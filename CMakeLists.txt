cmake_minimum_required(VERSION 4.1)
project(pca_relational_kd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# std::latch / std::jthread / std::counting_semaphore 在 GCC 9 上不存在。
if(NOT MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR
    "需要 GCC >= 11（std::latch / std::jthread / counting_semaphore）。\n"
    "当前编译器：${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# ---------------- libtorch ----------------
if(WIN32)
    set(LIBTORCH_ROOT "D:/DATA/C++_lib/libtorch_cu126")
else()
    set(LIBTORCH_ROOT "/home/chenzhijiang/userFile/C++_lib/libtorch")
    set(CUDA_TOOLKIT_ROOT_DIR "/home/chenzhijiang/userFile/miniConda3/envs/cuda126/targets/x86_64-linux")
    set(CMAKE_CUDA_FLAGS "-I${CUDA_TOOLKIT_ROOT_DIR}/include" CACHE STRING "CUDA flags")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
find_package(Torch REQUIRED)
message(STATUS "libtorch: ${TORCH_INSTALL_PREFIX}   (${Torch_VERSION})")

# ---------------- libjpeg-turbo ----------------
if(WIN32)
    set(LIBJPEG_TURBO_ROOT "D:/DATA/C++_lib/libjpeg-turbo")# 用导入库 jpeg.lib ⇒ 运行期需要 jpeg62.dll，见下方 POST_BUILD
else()
    set(LIBJPEG_TURBO_ROOT "/home/chenzhijiang/userFile/C++_lib/libjpeg-turbo")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LIBJPEG_TURBO_ROOT}")   # 必须在 find_package 之前
find_package(JPEG REQUIRED)

# Linux 上 std::thread/mutex/condition_variable 建在 pthread 上，需要 -pthread。
# Windows 上 Threads::Threads 展开为空，加了无害。
set(THREADS_PREFER_PTHREAD_FLAG ON)   # 否则可能只给 -lpthread，丢掉编译期的 _REENTRANT
find_package(Threads REQUIRED)
message(STATUS "JPEG include: ${JPEG_INCLUDE_DIR}")
message(STATUS "JPEG library: ${JPEG_LIBRARY}")

# ---------------- 源文件 ----------------
# 手写清单。CMake 不扫目录 —— 新加文件必须在这里补一行，否则不会进编译。
set(APP_SOURCES
    "src/pca_relational_kd.h"
    "src/pca_relational_kd.cpp"
    "src/config.h"               
    "src/config.cpp"
    "src/data/imagenet_loader.h" 
    "src/data/imagenet_loader.cpp"
    "src/data/transform.h"
    "src/data/transform.cpp"
    "src/data/decode.h"
    "src/data/decode.cpp"
    # "src/kd/loss.h"
    # "src/train/batch.h"
    # "src/data/cifar100.h"        "src/data/cifar100.cpp"
    # "src/data/transforms.h"      "src/data/transforms.cpp"
    # "src/model/resnet.h"         "src/model/resnet.cpp"
    # "src/model/checkpoint.h"     "src/model/checkpoint.cpp"
    # "src/train/trainer.h"        "src/train/trainer.cpp"
    # 方法本体：CoordinateSystem（fit / project / save / load）+ RKD-D / RKD-A
    # "src/kd/pca_rkd.h"           "src/kd/pca_rkd.cpp"
    )
add_executable(pca_relational_kd ${APP_SOURCES})

# ---------------- 头文件路径与链接 ----------------
target_include_directories(pca_relational_kd PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(pca_relational_kd PRIVATE torch JPEG::JPEG Threads::Threads)
# ---------------- 编译器选项（仅 MSVC）----------------
if(MSVC)
  target_compile_options(pca_relational_kd PRIVATE /bigobj /EHsc /wd4267 /wd4244 /utf-8)
  target_compile_definitions(pca_relational_kd PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------- 运行期依赖 ----------------
if(WIN32)
  # ⚠ 实测每个 build 目录 +3.86 GB（不是 1.5-2 GB）。所以只切 Release，别多开构建目录。
  file(GLOB TORCH_DLLS CONFIGURE_DEPENDS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pca_relational_kd POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} "$<TARGET_FILE_DIR:pca_relational_kd>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LIBJPEG_TURBO_ROOT}/bin/jpeg62.dll" "$<TARGET_FILE_DIR:pca_relational_kd>"
    COMMENT "拷贝 libtorch + libjpeg 运行期 DLL")
else()
  # Windows 靠拷 DLL，Linux 靠 RPATH。
  #   · libtorch/lib：DT_RPATH 对间接依赖也生效，libcudart.so.12 靠它找到
  #   · libjpeg 的 lib：否则运行时找不到 libjpeg.so
  #   · conda 的 lib：若用 conda 的 gcc 编译，它要 GLIBCXX_3.4.30+，系统那份没有
  get_filename_component(_JPEG_LIB_DIR "${JPEG_LIBRARY}" DIRECTORY)
  set(_PRKD_RPATH "${TORCH_INSTALL_PREFIX}/lib;${_JPEG_LIB_DIR}")
  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _PRKD_RPATH "$ENV{CONDA_PREFIX}/lib")
  endif()
  set_target_properties(pca_relational_kd PROPERTIES
    BUILD_RPATH   "${_PRKD_RPATH}"
    INSTALL_RPATH "${_PRKD_RPATH}")
endif()