// When $TRTEXEC_LOG_FILE is set, redirect stdout and stderr to the specified
// file as well.
#include <iostream>
#include <streambuf>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <stdlib.h>

namespace {
static struct redirect {
	std::streambuf *original_out = nullptr, *original_err = nullptr;
	class teebuf: public std::streambuf {
		public:
			teebuf(std::streambuf *a, std::streambuf *b): s1(a), s2(b) {}
		private:
			std::streambuf *s1, *s2;

			virtual int overflow(int c) override {
				if (c == EOF)
					return EOF;
				else {
					int r1 = s1->sputc(c);
					int r2 = s2->sputc(c);
					return (r1 == EOF || r2 == EOF) ? EOF : c;
				}
			}

			virtual int sync() override {
				int r1 = s1->pubsync();
				int r2 = s2->pubsync();
				return (r1 == 0 && r2 == 0) ? 0 : -1;
			}
	};
	redirect() {
		const char *fn = getenv("TRTEXEC_LOG_FILE");
		if (fn) {
			static std::ofstream ofs(std::filesystem::u8path(fn), std::ios::app);
			if (!ofs) throw std::runtime_error("Cannot open TRTEXEC_LOG_FILE");
			static teebuf out(ofs.rdbuf(), std::cout.rdbuf());
			static teebuf err(ofs.rdbuf(), std::cerr.rdbuf());
			original_out = std::cout.rdbuf(&out);
			original_err = std::cerr.rdbuf(&err);
		}
	}
	~redirect() {
		if (original_out) { std::cout.flush(); std::cout.rdbuf(original_out); }
		if (original_err) { std::cerr.flush(); std::cerr.rdbuf(original_err); }
	}
} _;
} // namespace
