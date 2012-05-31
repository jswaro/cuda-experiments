#include "levenshtein.h"
#include <cuda.h>
#include <math.h>

__device__ int __min(int a, int b) {
    return ((a)-(((a)-(b))&((b)-(a))>>31));
}

//N must be the size of the array
__device__ int __index(int i , int j, int n)
{
	int rval;
	if(!(i >= 0 && i < (n) && j >= 0 && j < (n) ) ) {
		rval = 0;
	} else if((i+j) < (n)) {
                rval = (((i+j)*(i+j+1))/2) + j;
	} else {
		rval = ((n)*(n)) -
                        (((2*(n) - (i+j))*(2*(n) - (i+j+1)))/2) +
			(j - ((j+i) - (n))) - 1;
	}
	return rval;
}

__global__ void levenshteinKernel(char* Md, char* Nd, int* Rd, int size) {
    __shared__ char Nds[ARRSIZE];   //Shared Nd character memory
    __shared__ int  Rs[ARRSIZE];    //Shared current min value memory
    int col = threadIdx.x + 1;      //column
    int row;                        //row
    char Mdt = Md[threadIdx.x];     //Character for this column

    Rd[0] = 0;

    Rd[__index(0, col,size)] = col;
    Rd[__index(col, 0,size)] = col;
    
    Nds[threadIdx.x]   = Nd[threadIdx.x];
    Rs[threadIdx.x]    = Rd[__index(0,col,size)];
    __syncthreads();

    
    for(int k = 2; k < (2 * size) + 1; ++k) {
        row = k - threadIdx.x;
        if( row > 0 && row <= size && col > 0 && col <= size )
        {
            Rs[threadIdx.x]       = __min(
                    (Rd[__index(row-1,col,size)] + 1),
                    (Rd[__index(row,col-1,size)] + 1 ) );
            Rd[__index(row,col,size)]  = __min(
                    (Rs[threadIdx.x]),
                    (Rd[__index(row-1,col-1,size)] + ((Mdt!=Nds[row-1])&1)) );
        }
        __syncthreads();
    }
}

__host__ void levenshteinCuda(char* s1, char* s2, int* &result, size_t size) {
    //Assumption is made that the size is a multiple of tile size
    dim3 dimGrid(1, 1);
    dim3 dimBlock(size, 1);
    
    char* Sd;
    char* Td;
    int*  Rd;
    size_t arrSize = (size+1) * (size+1);
    Sd = Td = NULL;
    Rd = NULL;

    cudaMalloc((void**) &Sd, (size *   sizeof(char)));
    cudaMalloc((void**) &Td, (size *   sizeof(char)));
    cudaMalloc((void**) &Rd, (arrSize *    sizeof(int)));

    cudaMemcpy(Sd, s1,     (size * sizeof(char)), cudaMemcpyHostToDevice);
    cudaMemcpy(Td, s2,     (size * sizeof(char)), cudaMemcpyHostToDevice);
    cudaMemset(Rd, 0, arrSize * sizeof(int));

    levenshteinKernel<<<dimGrid, dimBlock>>>(Sd,Td,Rd,size+1);

    cudaMemcpy(result, Rd, (arrSize * sizeof(int)), cudaMemcpyDeviceToHost);

    cudaFree(Sd);
    cudaFree(Td);
    cudaFree(Rd);
    return;
}

__host__ int getIndex(int row , int col)
{
    return ((row * (ARRSIZE + 1)) + col);
}

__host__ int getMin(int a, int b)
{
    return (a-((a-b)&(b-a)>>31));
}

