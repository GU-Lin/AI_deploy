cmake_minimum_required(VERSION 3.16)

project("main") # 建立一個CMake專案

# TensorRT 配置
set(TENSORRT_DIR "build_x86_64/TensorRT")
include_directories("${TENSORRT_DIR}/include")
include_directories(/"${TENSORRT_DIR}/include/common")
include_directories(/"${TENSORRT_DIR}/include/utils")
link_directories("${TENSORRT_DIR}/lib")
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)

# OpenCV 配置
find_package(OpenCV REQUIRED)

# CUDA 配置
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include 和 Source 資料夾
include_directories(${PROJECT_SOURCE_DIR}/include) # 添加 include 資料夾
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp") # 添加 src 資料夾內所有 .cpp 檔案
set(SOURCES ${SOURCES} main.cpp)
# 添加執行檔目標
add_executable(main  ${SOURCES}) # 建立目標，包含 src 資料夾中的所有 .cpp

# 目標的 include 路徑和所需的庫
target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
