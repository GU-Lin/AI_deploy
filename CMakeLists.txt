cmake_minimum_required(VERSION 3.16) # CMake_最低版本_需求

project("main") # 建立一個CMake專案，並給予專案名稱


set(TENSORRT_DIR "build_x86_64/TensorRT")
include_directories("${TENSORRT_DIR}/include")
link_directories("${TENSORRT_DIR}/lib")

set(OPENCV_DIR "build_x86_64/opencv/lib/cmake/opencv4")
include_directories("${OPENCV_DIR}/include")
link_directories("${OPENCV_DIR}/lib")
find_package(OpenCV REQUIRED)


find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
add_executable(main) # 建立一個CMake目標，目標為可執行檔

target_sources(main PRIVATE "main.cpp") # 指定建置該CMake目標時所使用的來源檔案
target_link_libraries(main ${TENSORRT_LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBS})
