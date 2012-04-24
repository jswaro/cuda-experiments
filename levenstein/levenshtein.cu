#include "levenshtein.h"
#include <cuda.h>
#include <math.h>

#define index(i,j) ((i * ARRSIZE) + j)


__global__ void levenshteinKernel(char* Md, char* Nd, int* Rd, int size) {
    __shared__ char Mds[ARRSIZE];   //Shared Md character memory
    __shared__ char Nds[ARRSIZE];   //Shared Nd character memory
    __shared__ int  Rs[ARRSIZE];    //Shared current min value memory
    __shared__ int  Rprev[ARRSIZE]; //Shared top value memory

    Mds[threadIdx.x]   = Md[threadIdx.x];
    Nds[threadIdx.x]   = Nd[threadIdx.x];
    Rs[threadIdx.x]    = Rd[threadIdx.x];
    Rprev[threadIdx.x] = Rs[threadIdx.x];

    __syncthreads();

    int i = threadIdx.x + 1;  //column
    int j;                    //row

    for(int k = 0; k < (2 * size) - 1; ++k) {
        j = k - threadIdx.x;
        if( j >=0 && j < size)
        {
            Rs[threadIdx.x] = MIN( (Rd[index(j, i - 1)] + 1),
                                   (Rprev[threadIdx.x] + 1) );
            Rd[index(j,i)]  = MIN( (Rs[threadIdx.x]),
                                   (Rd[index(j-1,i-1)] + ((Mds[i-1]==Nds[j-1])&1)) );
        }

        __syncthreads();
        Rprev[threadIdx.x] = Rs[threadIdx.x];
        __syncthreads();
    }
    
    
}

__host__ void levenshteinCuda(char* s1, char* s2, int** &result,
        size_t size) {
    //Assumption is made that the size is a multiple of tile size
    dim3 dimGrid(1, 1);
    dim3 dimBlock(ARRSIZE, 1);
    
    char* Sd;
    char* Td;
    int*  Rd;
    size_t arrSize = ARRSIZE * ARRSIZE;
    Sd = Td = NULL;
    Rd = NULL;

    for (int i = 0; i < ARRSIZE + 1; i++)
        result[i][0] = i;

    for (int i = 0; i < ARRSIZE + 1; i++)
        result[0][i] = i;

    cudaMalloc((void**) &Sd, (size *   sizeof(char)));
    cudaMalloc((void**) &Td, (size *   sizeof(char)));
    cudaMalloc((void**) &Rd, (arrSize *    sizeof(int)));

    cudaMemcpy(Sd, s1,     (size * sizeof(char)), cudaMemcpyHostToDevice);
    cudaMemcpy(Td, s2,     (size * sizeof(char)), cudaMemcpyHostToDevice);
    cudaMemcpy(Rd, result, (arrSize * sizeof(int)),  cudaMemcpyHostToDevice);
   
    levenshteinKernel<<<dimGrid, dimBlock>>>(Sd,Td,Rd,size);

    cudaMemcpy(result, Rd, (arrSize * sizeof(size_t)), cudaMemcpyDeviceToHost);

    cudaFree(Sd);
    cudaFree(Td);
    cudaFree(Rd);

    return;
}