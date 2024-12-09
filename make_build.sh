#!/bin/bash
engine="TensorRT"
arch="x86_64"
cuda_version="12.6"
ubuntu_version="22.04"

make_TensorRT()
{
    cd 3rdparty/TensorRT
    ./docker/build.sh --file docker/ubuntu-${ubuntu_version}.Dockerfile --tag tensorrt-ubuntu${ubuntu_version}-cuda${cuda_version}:${arch}
    cd ../../

    docker run --rm -it -v `pwd`:/workspace tensorrt-ubuntu22.04-cuda12.6 bash -c \
    "mkdir build_${arch} && cd build_${arch} && mkdir ${engine} &&
    cp -rf /TensorRT*/include /workspace/build_${arch}/${engine} &&
    cp -rf /TensorRT*/targets/*/bin /workspace/build_${arch}/${engine} &&
    cp -rf /TensorRT*/targets/*/lib /workspace/build_${arch}/${engine} &&
    exit
    "
}

make_opencv()
{
    cd docker
    docker build -t opencv -f build_opencv.dockerfile .
    cd ..
    docker run --rm -it -v `pwd`:/workspace opencv bash -c \
    "cp -rf /opencv /workspace/build_${arch} && \
    exit"

}
# make_${engine}
make_opencv
