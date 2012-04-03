#include "matrix.h"
#include <cuda.h>

__global__ void matrixMulKernel(float* Md, float* Nd, float* Rd, int size) {
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0;

    for(int i = 0; i < size; ++i) {
        sum += Md[row * size + i] * Nd[i * size + col];
    }
    Rd[row * size + col] = sum;
    
}

__host__ void matrixMulCuda(float* m1, float* m2, float* &result,
        size_t size) {
    //Assumption is made that the size is a multiple of tile size
    dim3 dimGrid((size/TILE_SIZE), (size/TILE_SIZE));
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    
    float* Md;
    float* Nd;
    float* Rd;
    size_t arrSize = size * size * sizeof(float);
    Md = Nd = Rd = NULL;

    cudaMalloc((void**) &Md, arrSize);
    cudaMalloc((void**) &Nd, arrSize);
    cudaMalloc((void**) &Rd, arrSize);

    cudaMemcpy(Md, m1, arrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, m2, arrSize, cudaMemcpyHostToDevice);

    for(int i = 0; i < 1000; ++i)
        matrixMulKernel<<<dimGrid,dimBlock>>>(Md, Nd, Rd, size);

    cudaMemcpy(result, Rd, arrSize, cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Rd);

    return;
}