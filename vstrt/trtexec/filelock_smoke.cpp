// Exercise the actual patched SDK file-lock helper without a GPU or engine.
#include <filesystem>
#include <iostream>
#if VSMLRT_TRT_MAJOR == 8
#include "common.h"
#else
#include "utils/fileLock.h"
class QuietLogger : public nvinfer1::ILogger {
    void log(Severity, char const*) noexcept override {}
};
#endif

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string filename(argv[1]);
    const auto lockPath = std::filesystem::u8path(filename + ".lock");
    try {
        {
#if VSMLRT_TRT_MAJOR == 8
            samplesCommon::FileLock lock(filename);
#else
            QuietLogger logger;
            nvinfer1::utils::FileLock lock(logger, filename);
#endif
            if (!std::filesystem::exists(lockPath)) return 3;
        }
        if (std::filesystem::exists(lockPath)) return 4;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 5;
    }
    std::cout << "UTF-8 file lock created and removed\n";
    return 0;
}
