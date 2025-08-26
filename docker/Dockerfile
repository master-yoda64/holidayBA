# syntax = docker/dockerfile:1
# FROM ubuntu:24.04
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
SHELL ["/bin/bash", "-c"]

# apt-get
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.iij.ad.jp/pub/linux/ubuntu/archive/%g" /etc/apt/sources.list
#タイムゾーン
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

# apt-get更新 wget取得
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y wget apt-utils curl less nano software-properties-common \
&& apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# 日本語環境
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales language-pack-ja-base language-pack-ja fonts-ipafont-gothic \
&& locale-gen ja_JP.UTF-8 \
&& echo 'export LANG=ja_JP.UTF-8' >> /root/.bashrc \
&& source /root/.bashrc

ENV PYTHONIOENCODING "utf-8"
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

# .bashrc更新
RUN source /root/.bashrc

#tmpディレクトリ作成
RUN mkdir /root/tmp
WORKDIR /root/tmp/

#===================================#
# ssh setup                         #
#===================================#
WORKDIR /root
RUN apt update && apt install openssh-client openssh-server -y
RUN --mount=type=secret,id=ssh_key,dst=/root/.ssh/id_rsa \
    mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

#===================================#
# python and pip installation       #
#===================================#

ENV PYTHONIOENCODING "utf-8"
RUN apt-get install -y python3-pip python3-dev python3-setuptools

#===================================#
# get dependencies                  #
#===================================#

RUN apt-get update && apt-get upgrade -y \
&&  apt-get install -y unzip libomp-dev libglew-dev libqhull* \
    libusb-1.0-0-dev libusb-dev libgtest-dev git-core freeglut3-dev pkg-config build-essential libflann-dev libflann1.9 \
    libpthread-stubs0-dev libboost-all-dev texinfo ffmpeg libgtk-3-dev libjpeg-dev libpng-dev \
    libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libv4l-dev qtbase5-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev x264 v4l-utils libglib2.0-0 libsm6 libxext6 libxrender-dev openni2-utils libopenni2-dev \
    flake8 python3-numpy python3-flake8 libleveldb-dev liblmdb-dev libatlas-base-dev liblapack-dev \
&& apt-get update &&  apt-get upgrade -y && apt-get autoremove -y
RUN apt install liblapacke-dev libtbb-dev qv4l2 libsuitesparse-dev libgoogle-glog-dev libgflags-dev libgles2-mesa-dev libhdf5-dev -y
#===================================#
# opencv installaitiom              #
#===================================#
RUN apt install -y cmake
ENV INSTALL_OPENCV_VERSION "4.10.0"
WORKDIR /root/tmp

RUN wget https://github.com/opencv/opencv/archive/${INSTALL_OPENCV_VERSION}.tar.gz
RUN tar xvf ${INSTALL_OPENCV_VERSION}.tar.gz
RUN rm -r ${INSTALL_OPENCV_VERSION}.tar.gz
RUN wget https://github.com/opencv/opencv_contrib/archive/${INSTALL_OPENCV_VERSION}.tar.gz
RUN tar xvf ${INSTALL_OPENCV_VERSION}.tar.gz
RUN rm -r ${INSTALL_OPENCV_VERSION}.tar.gz
WORKDIR /root/tmp/opencv-${INSTALL_OPENCV_VERSION}/build

# CUDA_ARCH_BINについて
# nvidiaのboardはarchtecture numberが振られているので、それを指定する
# 参考: https://qiita.com/k_ikasumipowder/items/1142dadba01b42ac6012
# 8.6はRTX 3060のarchtecture number

RUN cmake \
    -D WITH_OPENMP=ON \
    -D WITH_CUDA=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D PYTHON_EXECUTABLE=$(which python3) \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/root/tmp/opencv_contrib-${INSTALL_OPENCV_VERSION}/modules \
    -D WITH_GSTREAMER=ON \
    -D WITH_LIBV4L=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D CMAKE_BUILD_TYPE=RELEASE .. \
&& make -j10 && make install && /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && ldconfig
RUN apt install -y libeigen3-dev
WORKDIR /root

#===================================#
# ceres solver installaitiom        #
#===================================#
WORKDIR /root/tmp
RUN git clone https://ceres-solver.googlesource.com/ceres-solver -b 2.2.0
RUN apt-get install libgoogle-glog-dev libgflags-dev -y
RUN apt-get install libatlas-base-dev -y
RUN apt-get install libeigen3-dev -y
WORKDIR /root/tmp/ceres-solver
RUN mkdir ceres-bin
WORKDIR /root/tmp/ceres-solver/ceres-bin
RUN cmake \
    -DUSE_CUDA=OFF \
    -DSUITESPARSE=ON \
    -DBUILD_DOCUMENTATION=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE=-O3 ..
RUN make -j12
#RUN make test
RUN make install

#===================================#
# sophus installaitiom              #
#===================================#
WORKDIR /root/tmp
RUN git clone https://github.com/strasdat/Sophus.git
WORKDIR /root/tmp/Sophus
RUN mkdir build
WORKDIR /root/tmp/Sophus/build
RUN cmake  ..
RUN make -j12
RUN make install

WORKDIR /root
RUN ln -sf eigen3/Eigen Eigen