# Kmeans CUDA C/C++
Parallel K-means Clustering using CUDA C/C++
## Description
Parallelizing K-means Clustering algorithm using CUDA C/C++ with CIFAR-10 dataset.
## Getting Started
### Executing program
* Compiling with nvcc
```
nvcc -O3 kmeans_parallel.cu -o kmeans_parallel
./kmeans_parallel
```