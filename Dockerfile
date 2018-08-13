FROM ubuntu:xenial

ENV INSTALL_PREFIX /usr/local
ENV LD_LIBRARY_PATH ${INSTALL_PREFIX}/lib

#Preparation
RUN apt-get update && apt-get install -y \
  curl \
  wget \
  git \
  gcc \
  g++ \
  cmake \
  cmake-data \
  libopencv-dev \
  protobuf-compiler \
  libprotobuf-dev \
  ruby-dev \
  ruby-rmagick \
  ruby-bundler && \
  rm -rf /var/lib/apt/lists/*

# MKL-DNN
RUN mkdir /opt/mkl-dnn
WORKDIR /opt/mkl-dnn
RUN git clone https://github.com/01org/mkl-dnn.git && \
    cd mkl-dnn/scripts && bash ./prepare_mkl.sh && cd .. && \
    sed -i 's/add_subdirectory(examples)//g' CMakeLists.txt && \
    sed -i 's/add_subdirectory(tests)//g' CMakeLists.txt && \
    mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. && make && \
    make install

# Menoh
WORKDIR /opt/
RUN git clone https://github.com/pfnet-research/menoh.git && \
    cd menoh && \
    sed -i 's/add_subdirectory(example)//g' CMakeLists.txt && \
    sed -i 's/add_subdirectory(test)//g' CMakeLists.txt && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. && \
    make install

# Install Node.js
RUN apt-get -qq update
RUN apt-get install -y nodejs npm
RUN npm install n -g
RUN n stable
RUN ln -sf /usr/local/bin/node /usr/bin/node
RUN apt-get purge -y nodejs npm

# Create program execution environment
RUN mkdir -p /usr/src/app
COPY . /usr/src/app/
WORKDIR /usr/src/app
RUN npm install
CMD node /usr/src/app/example_mnist.js
