cmake_minimum_required(VERSION 3.20)

project(vs-trt VERSION 3.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
set(TENSORRT_HOME "" CACHE PATH "Path to TensorRT")
option(USE_NVINFER_PLUGIN "Initialize nvinfer_plugin" FALSE)
option(USE_NVINFER_PLUGIN_STATIC "Use static nvinfer_plugin" FALSE)
set(TENSORRT_LIBRARY_SUFFIX "" CACHE STRING "TensorRT library suffix")

set(TENSORRT_LIBRARY_DIRS
    "${TENSORRT_HOME}/lib"
    "${TENSORRT_HOME}/targets/x86_64-linux-gnu/lib"
)

if (WIN32)
    file(GLOB files "${TENSORRT_HOME}/lib/tensorrt_rtx*.lib")
else()
    file(GLOB files "${TENSORRT_HOME}/lib/libtensorrt_rtx*.so")
endif()
if (files)
    set(USE_TRT_RTX TRUE)
else()
    set(USE_TRT_RTX FALSE)
endif()

FIND_PACKAGE(CUDAToolkit REQUIRED)

if (USE_TRT_RTX)
    add_library(vstrt_rtx SHARED
        $<$<PLATFORM_ID:Windows>: longpath.manifest>
        vs_tensorrt.cpp
        win32.cpp
    )

    target_include_directories(vstrt_rtx PRIVATE
        ${VAPOURSYNTH_INCLUDE_DIRECTORY}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_HOME}/include
    )

    set_target_properties(vstrt_rtx PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    if (UNIX AND NOT APPLE)
        set_target_properties(vstrt_rtx PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )
    elseif (APPLE)
        set_target_properties(vstrt_rtx PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path"
        )
    endif()

    target_link_directories(vstrt_rtx PRIVATE ${TENSORRT_HOME}/lib)
    target_link_libraries(vstrt_rtx PRIVATE CUDA::cudart_static)

    if (WIN32)
        target_link_libraries(vstrt_rtx PRIVATE "tensorrt_rtx${TENSORRT_LIBRARY_SUFFIX}")
        if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_link_options(vstrt_rtx PRIVATE "/DELAYLOAD:tensorrt_rtx${TENSORRT_LIBRARY_SUFFIX}.dll" "delayimp.lib")
        endif()
    else()
        target_link_libraries(vstrt_rtx PRIVATE tensorrt_rtx)
    endif()

    target_compile_definitions(vstrt_rtx PRIVATE USE_NVINFER_PLUGIN)

    target_include_directories(vstrt_rtx PUBLIC
        "${PROJECT_BINARY_DIR}"
    )

    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VCS_TAG
    )
    string(STRIP ${VCS_TAG} VCS_TAG)
    configure_file(config.h.in config.h)

    install(TARGETS vstrt_rtx
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
else()
    add_library(vstrt SHARED
        $<$<PLATFORM_ID:Windows>: longpath.manifest>
        vs_tensorrt.cpp
        win32.cpp
    )

    target_include_directories(vstrt PRIVATE
        ${VAPOURSYNTH_INCLUDE_DIRECTORY}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_HOME}/include
    )

    set_target_properties(vstrt PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    if (UNIX AND NOT APPLE)
        set_target_properties(vstrt PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )
    elseif (APPLE)
        set_target_properties(vstrt PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path"
        )
    endif()

    target_link_directories(vstrt PRIVATE ${TENSORRT_LIBRARY_DIRS})
    if (WIN32)
        set(TENSORRT_RUNTIME_LIBRARY "nvinfer${TENSORRT_LIBRARY_SUFFIX}")
    else()
        # Runtime-only TensorRT archives often omit the development symlink
        # libnvinfer.so. Resolve the versioned ELF directly for Linux fallback.
        file(GLOB TENSORRT_RUNTIME_CANDIDATES
            "${TENSORRT_HOME}/lib/libnvinfer.so"
            "${TENSORRT_HOME}/lib/libnvinfer.so.*"
            "${TENSORRT_HOME}/targets/x86_64-linux-gnu/lib/libnvinfer.so"
            "${TENSORRT_HOME}/targets/x86_64-linux-gnu/lib/libnvinfer.so.*"
        )
        list(SORT TENSORRT_RUNTIME_CANDIDATES)
        list(LENGTH TENSORRT_RUNTIME_CANDIDATES TENSORRT_RUNTIME_COUNT)
        if (TENSORRT_RUNTIME_COUNT EQUAL 0)
            message(FATAL_ERROR "Could not find libnvinfer.so* under ${TENSORRT_HOME}/lib")
        endif()
        list(GET TENSORRT_RUNTIME_CANDIDATES -1 TENSORRT_RUNTIME_LIBRARY)
    endif()
    target_link_libraries(vstrt PRIVATE CUDA::cudart_static "${TENSORRT_RUNTIME_LIBRARY}")

    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_link_options(vstrt PRIVATE "/DELAYLOAD:nvinfer${TENSORRT_LIBRARY_SUFFIX}.dll" "delayimp.lib")
    endif()

    if (USE_NVINFER_PLUGIN)
        add_definitions(-DUSE_NVINFER_PLUGIN)
        if (USE_NVINFER_PLUGIN_STATIC)
            target_link_libraries(vstrt PRIVATE "nvinfer_plugin_static${TENSORRT_LIBRARY_SUFFIX}")
        else()
            target_link_libraries(vstrt PRIVATE "nvinfer_plugin${TENSORRT_LIBRARY_SUFFIX}")

            if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
                target_link_options(vstrt PRIVATE "/DELAYLOAD:nvinfer_plugin${TENSORRT_LIBRARY_SUFFIX}.dll")
            endif()
        endif()
    endif()

    target_include_directories(vstrt PUBLIC
        "${PROJECT_BINARY_DIR}"
    )

    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VCS_TAG
    )
    string(STRIP ${VCS_TAG} VCS_TAG)
    configure_file(config.h.in config.h)

    install(TARGETS vstrt
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

