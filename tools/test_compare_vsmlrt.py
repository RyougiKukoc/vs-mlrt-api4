"""Regression tests for comparison verdicts; no GPU or VapourSynth needed."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_api3_api4_vsmlrt_backends as compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def reports(self, left, right):
        reports = {}
        for env, value in (("api3", left), ("api4", right)):
            path = self.root / f"{env}.npz"
            np.savez(path, p0=np.asarray(value, dtype=np.float32).reshape(1, -1))
            reports[env] = {
                "ok": True, "width": 2, "height": 1, "format": "GrayS",
                "array_path": str(path),
            }
        return reports

    def parent(self, reports, atol=1e-5):
        args = argparse.Namespace(
            output_dir=self.root, api3_python=Path(sys.executable),
            api4_python=Path(sys.executable), width=2, height=1, atol=atol, timeout=10,
        )
        with patch.object(compare, "BACKENDS", ("fixture",)), \
                patch.object(compare, "MODELS", ("fixture",)), \
                patch.object(compare, "run_worker_case", side_effect=lambda **kw: reports[kw["env_name"]]), \
                contextlib.redirect_stdout(io.StringIO()):
            code = compare.run_parent(args)
        contents = (self.root / "summary.json").read_text(encoding="utf-8")
        summary = json.loads(contents, parse_constant=lambda value: self.fail(f"invalid JSON number: {value}"))
        return code, summary

    def test_exact_and_tolerated_outputs_pass(self):
        for output, status in (([1, 2], "EXACT"), ([1.000001, 2], "CLOSE")):
            with self.subTest(output=output):
                code, summary = self.parent(self.reports([1, 2], output))
                self.assertEqual(code, 0)
                self.assertTrue(summary["ok"])
                self.assertEqual(summary["failed_cases"], 0)
                self.assertTrue(summary["cases"][0]["status"].startswith(status))

    def test_worker_failure_and_excessive_difference_fail_parent(self):
        reports = self.reports([1, 2], [2, 2])
        for candidate in (reports, {"api3": reports["api3"], "api4": {"ok": False}}):
            with self.subTest(candidate=candidate):
                code, summary = self.parent(candidate)
                self.assertEqual(code, 1)
                self.assertFalse(summary["ok"])
                self.assertEqual(summary["failed_cases"], 1)
                self.assertFalse(summary["cases"][0]["accepted"])

    def test_nonfinite_output_is_rejected_even_when_both_match(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for left in ([1, 2], [value, 2]):
                with self.subTest(left=left, value=value):
                    code, summary = self.parent(self.reports(left, [value, 2]))
                    self.assertEqual(code, 1)
                    self.assertFalse(summary["ok"])
                    self.assertIn("NaN or infinity", summary["cases"][0]["comparison"]["reason"])

    def test_missing_or_corrupt_arrays_fail_with_a_report(self):
        reports = self.reports([1, 2], [1, 2])
        path = Path(reports["api4"]["array_path"])
        path.unlink()
        code, summary = self.parent(reports)
        self.assertEqual(code, 1)
        self.assertIn("could not read", summary["cases"][0]["comparison"]["reason"])
        path.write_bytes(b"not an array archive")
        self.assertEqual(self.parent(reports)[0], 1)
        reports = self.reports([1, 2], [1, 2])
        path.write_bytes(path.read_bytes()[:-24])
        code, summary = self.parent(reports)
        self.assertEqual(code, 1)
        self.assertFalse(summary["ok"])
        self.assertIn("could not read", summary["cases"][0]["comparison"]["reason"])

    def test_signed_zero_is_close_but_not_byte_exact(self):
        code, summary = self.parent(self.reports([0.0, 1], [-0.0, 1]), atol=0)
        self.assertEqual(code, 0)
        self.assertTrue(summary["ok"])
        self.assertFalse(summary["cases"][0]["comparison"]["exact"])
        self.assertTrue(summary["cases"][0]["status"].startswith("CLOSE"))

    def test_no_old_report_or_array_can_mask_a_silent_worker(self):
        path = self.root / "api4" / "ncnn" / "dpir.json"
        compare.write_json(path, {"ok": True, "from_previous_run": True})
        path.with_suffix(".npz").write_bytes(b"stale")
        with patch.object(compare, "make_worker_command", return_value=[sys.executable, "-c", "pass"]):
            report = compare.run_worker_case(Path(sys.executable), "api4", "ncnn", "dpir", self.root, 2, 1, 10)
        self.assertEqual(report["worker_returncode"], 0)
        self.assertFalse(report["ok"])
        self.assertNotIn("from_previous_run", report)
        self.assertFalse(path.with_suffix(".npz").exists())

    def test_timeout_output_and_launch_error_are_serializable_failures(self):
        error = subprocess.TimeoutExpired("worker", 1, output=b"partial output", stderr=b"diagnostic")
        for failure in (error, OSError("cannot launch interpreter")):
            with self.subTest(failure=failure), patch.object(compare.subprocess, "run", side_effect=failure):
                report = compare.run_worker_case(Path(sys.executable), "api4", "ncnn", "dpir", self.root, 2, 1, 1)
                self.assertFalse(report["ok"])
                json.dumps(report, allow_nan=False)
                self.assertIn("error", report)

    def test_bad_worker_json_is_a_failure(self):
        path = self.root / "api4" / "ncnn" / "dpir.json"
        for payload in ("{broken", "[]", '{"ok": true, "metric": NaN}'):
            def run(*args, **kwargs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload, encoding="utf-8")
                return subprocess.CompletedProcess([], 0, "", "")
            with self.subTest(payload=payload), patch.object(compare.subprocess, "run", side_effect=run):
                report = compare.run_worker_case(Path(sys.executable), "api4", "ncnn", "dpir", self.root, 2, 1, 10)
                self.assertFalse(report["ok"])
                self.assertIn("invalid worker report", report["error"])

    def test_invalid_tolerance_is_rejected_by_cli(self):
        for tolerance in ("nan", "inf", "-1"):
            result = subprocess.run(
                [sys.executable, str(Path(compare.__file__)), f"--atol={tolerance}"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("finite and non-negative", result.stderr)

    def test_trt_rtx_is_an_explicitly_supported_backend(self):
        args = compare.parse_args(["--backends", "trt_rtx", "--models", "dpir"])
        self.assertEqual(args.backends, ["trt_rtx"])
        self.assertEqual(args.models, ["dpir"])


if __name__ == "__main__":
    unittest.main()
