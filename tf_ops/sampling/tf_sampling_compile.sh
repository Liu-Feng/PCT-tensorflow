#!/bin/bash
CUDA_ROOT=/usr/local/cuda-10.0
TF_ROOT=/home/liufeng/anaconda3/envs/tf131/lib/python3.6/site-packages/tensorflow

${CUDA_ROOT}/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -lcudart -L ${CUDA_ROOT}/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


###ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
