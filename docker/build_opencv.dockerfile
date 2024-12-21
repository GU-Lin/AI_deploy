# 使用 Ubuntu 作為基礎映像
FROM ubuntu:22.04

# 更新軟件包並安裝必要的依賴

RUN apt-get update && apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    echo "Asia/Taipei" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-pip

# 安裝 Python 相關依賴
RUN pip3 install numpy

# 克隆 OpenCV 代碼並構建
WORKDIR /root
RUN git clone --branch 4.x https://github.com/opencv/opencv.git

WORKDIR /root/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/opencv \
          -D BUILD_EXAMPLES=ON ..
RUN make -j 10
RUN make install

# 驗證安裝
# RUN python3 -c "import cv2; print(cv2.__version__)"

# 設置工作目錄
WORKDIR /workspace
