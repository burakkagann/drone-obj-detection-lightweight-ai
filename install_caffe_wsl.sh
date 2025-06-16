#!/bin/bash

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    libopencv-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-opencv

# Install Python dependencies
pip3 install numpy protobuf

# Clone Caffe
cd ~
git clone https://github.com/BVLC/caffe.git
cd caffe

# Install Python dependencies
pip3 install -r python/requirements.txt

# Create build directory
mkdir build
cd build

# Configure and build Caffe
cmake ..
make -j$(nproc)
make install

# Add Caffe to Python path
echo 'export PYTHONPATH=/home/$USER/caffe/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

# Test installation
python3 -c "import caffe; print('Caffe installation successful!')" 