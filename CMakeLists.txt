cmake_minimum_required(VERSION 4.1)
project(pca_relational_kd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# ---------------- 源文件 ----------------
# 手写清单。CMake 不扫目录 —— 新加文件必须在这里补一行，否则不会进编译。
set(APP_SOURCES
    "src/pca_relational_kd.h"
    "src/pca_relational_kd.cpp"
    "src/config.h"               
    "src/config.cpp"
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
target_link_libraries(pca_relational_kd PRIVATE torch)

# ---------------- 编译器选项（仅 MSVC）----------------
if(MSVC)
  target_compile_options(pca_relational_kd PRIVATE /bigobj /EHsc /wd4267 /wd4244 /utf-8)
  target_compile_definitions(pca_relational_kd PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------- 运行期依赖 ----------------
if(WIN32)
  # ⚠ 这一步会给每个 build 目录塞一整套 libtorch DLL（~1.5-2 GB）。
  #   所以只切 Release，别 Debug 和 Release 各编一次。
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pca_relational_kd POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} "$<TARGET_FILE_DIR:pca_relational_kd>")
else()
  set_target_properties(pca_relational_kd PROPERTIES
    BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()