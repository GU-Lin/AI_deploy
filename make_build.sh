#!/bin/bash
engine="TensorRT"
arch="x86_64"
cuda_version="12.6"
ubuntu_version="22.04"

usage() {
    echo "Usage: $0 [--opencv | --tensorRT | --tensorflow | --all]"
    echo "Options:"
    echo "  --opencv       Build OpenCV"
    echo "  --tensorRT     Build TensorRT"
    echo "  --tensorflow   Build TensorFlow"
    echo "  --all          Build all components"
    exit 1
}

make_TensorRT()
{
    cd 3rdparty/TensorRT
    ./docker/build.sh --file docker/ubuntu-${ubuntu_version}.Dockerfile --tag tensorrt-ubuntu${ubuntu_version}-cuda${cuda_version}:${arch}
    cd ../../

    docker run --rm -it -v `pwd`:/workspace tensorrt-ubuntu22.04-cuda12.6:${arch} bash -c \
    "mkdir build_${arch} && cd build_${arch} && mkdir ${engine} &&
    cp -rf /TensorRT*/include /workspace/build_${arch}/${engine} &&
    cp -rf /TensorRT*/targets/*/bin /workspace/build_${arch}/${engine} &&
    cp -rf /TensorRT*/targets/*/lib /workspace/build_${arch}/${engine} &&
    cp -rf /TensorRT*/samples/common /workspace/build_${arch}/${engine}/include &&
    cp -rf /TensorRT*/samples/utils /workspace/build_${arch}/${engine}/include &&
    exit
    "
}

check_and_create_folder() {
    folder_name="$1"

    if [[ -d "$folder_name" ]]; then
        echo "Folder '$folder_name' exists"
    else
        echo "Folder '$folder_name' doesn't exist, making..."
        mkdir -p "$folder_name"
        echo "Folder '$folder_name' make done"
    fi
}

make_opencv()
{
    cd docker
    docker build -t opencv -f build_opencv.dockerfile .
    cd ..
    docker run --rm  -it -v `pwd`:/workspace opencv bash -c \
    "cp -rf /opencv /workspace/build_${arch} && exit"

}

make_tensorflow()
{
    cd docker
    docker build -t tensorflow -f build_tensorflow.dockerfile .
    cd ..
    docker run --rm  -it -v `pwd`:/workspace tensorflow bash -c \
    "cp -rf /tensorflow /workspace/build_${arch} && exit"

}

main() {
    # 檢查並建立 build_x86_64 資料夾
    check_and_create_folder "build_${arch}"

    # 建構指定套件
    case $1 in
        --opencv)
            make_opencv
            ;;
        --tensorRT)
            make_TensorRT
            ;;
        --tensorflow)
            make_tensorflow
            ;;
        --all)
            echo "Build all package ..."
            make_opencv
            make_TensorRT
            make_tensorflow
            echo "Build package done"
            ;;
        *)
            echo "Arg : $0 [--opencv | --tensorRT | --tensorflow | --all]"
            exit 1
            ;;
    esac
}

main "$@"
