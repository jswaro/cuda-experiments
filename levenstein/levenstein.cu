#include "levenstein.h"
#include <cuda.h>
#include <math.h>

__global__ void levensteinKernel(char* Md, char* Nd, size_t* Rd, int size) {
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0;

    for(int i = 0; i < size; ++i) {
        sum += Md[row * size + i] * Nd[i * size + col];
    }
    Rd[row * size + col] = sum;
    
}

__host__ void levensteinCuda(char* s1, char* s2, int* &result,
        size_t size) {
    //Assumption is made that the size is a multiple of tile size
    dim3 dimGrid((size/TILE_SIZE), (size/TILE_SIZE));
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    
    char* Sd;
    char* Td;
    size_t* Rd;
    size_t arrSize = size * size;
    Sd = Td = Rd = NULL;

    cudaMalloc((void**) &Sd, (arrSize *   sizeof(char)));
    cudaMalloc((void**) &Td, (arrSize *   sizeof(char)));
    cudaMalloc((void**) &Rd, (arrSize * sizeof(size_t)));

    cudaMemcpy(Sd, s1, (arrSize * sizeof(char)), cudaMemcpyHostToDevice);
    cudaMemcpy(Td, s2, (arrSize * sizeof(char)), cudaMemcpyHostToDevice);


    for( size_t i = 0; i < ((size - 1)  * 2) + 1; ++i ) 
    {
        size_t stripe_size = (size / 2) - abs( i - (size / 2);
        size_t numBlocks = 1;
        if( stripe_size > STRIPE_MAX )
        {
            numBlocks = (size_t) ceil(stripe_size / (float) STRIPE_MAX);
            size_t numWarps = (size_t) ceil(stripe_size / (float) WARP_MAX);

            size_t warpsPerBlock = numWarps / numBlocks; // in terms of warps
            int i;
            levensteinKernel<<<numBlocks, (warpsPerBlock * WARP_MAX)>>>(Sd, Td, Rd, size);



        } else {
            levensteinKernel<<<numBlocks,stripe_size>>>(Md, Nd, Rd, stripe_size);
        }

    }
       

    cudaMemcpy(result, Rd, (arrSize * sizeof(size_t)), cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Rd);

    return;
}