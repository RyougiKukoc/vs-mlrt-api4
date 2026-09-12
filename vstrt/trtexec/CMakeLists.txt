cmake_minimum_required(VERSION 3.24)
project(vsmlrt_trtexec LANGUAGES CXX)
include(GNUInstallDirs)

# Staged into the matching TensorRT OSS samples/trtexec tree. Public 8.6 and
# 11.1 samples have different entry points and common sources.
if(NOT VSMLRT_TRT_VERSION MATCHES "^(8\\.6\\.1|11\\.1\\.0)\\.")
    message(FATAL_ERROR "Unsupported custom trtexec SDK: ${VSMLRT_TRT_VERSION}")
endif()
if(NOT EXISTS "${TENSORRT_HOME}/include/NvInferVersion.h")
    message(FATAL_ERROR "TENSORRT_HOME must name the selected binary SDK")
endif()
find_package(CUDAToolkit REQUIRED)
set(common_sources
    ../common/logger.cpp
    ../common/sampleEngines.cpp
    ../common/sampleInference.cpp
    ../common/sampleOptions.cpp
    ../common/sampleReporting.cpp
    ../common/sampleUtils.cpp
)
set(extra_sources)
if(VSMLRT_TRT_VERSION MATCHES "^11\\.")
    enable_language(CUDA)
    list(APPEND extra_sources
        trtexec_main.cpp
        ../common/bigInt.cpp
        ../common/bfloat16.cpp
        ../common/common.cpp
        ../common/debugTensorWriter.cpp
        ../common/globalTimerKernel.cu
        ../common/sampleDevice.cpp
        ../common/sampleTuning.cpp
        ../../shared/utils/fileLock.cpp
        ../../shared/utils/cacheUtils.cpp
    )
    set(lock_sources ../../shared/utils/fileLock.cpp)
    set(trt_major 11)
else()
    # 8.6 has main() in trtexec.cpp and header-only device/file-lock helpers.
    set(lock_sources ../common/logger.cpp)
    set(trt_major 8)
endif()
add_executable(trtexec trtexec.cpp logfile.cpp ${common_sources} ${extra_sources})
add_executable(trtexec_filelock_smoke filelock_smoke.cpp ${lock_sources})
foreach(target trtexec trtexec_filelock_smoke)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target} PRIVATE TRT_STATIC=0 VSMLRT_TRT_MAJOR=${trt_major})
    target_include_directories(${target} PRIVATE
        "${TENSORRT_HOME}/include"
        ../common .. ../../include ../../shared ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE CUDA::cudart_static ${CMAKE_DL_LIBS})
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHsc;/utf-8>)
        target_sources(${target} PRIVATE trtexec.manifest)
    endif()
endforeach()
install(TARGETS trtexec trtexec_filelock_smoke RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
