#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include "matrix.h"
#include <cuda_runtime.h>
#include <math.h>

using namespace std;

void fillArray( float* arr, size_t row, size_t col ) {
    for(size_t x = 0; x < row; ++x)
    {
        for(size_t y = 0; y < col; ++y)
        {
            arr[x * col + y] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    return;
}

void getXYArray( float* &arr, size_t r, size_t c) {
    arr = new float[r * c];
    for(size_t x = 0; x < r; ++x) {
        for(size_t y = 0; y < c; ++y)
        {
            arr[x * c + y] = 0;
        }
    }
    return;
}

void printArray( float* arr, size_t r, size_t c) {
    string s;
    char val[1024];
    for(size_t x = 0; x < r; ++x)
    {
        for(size_t y = 0; y < c; ++y)
        {
            sprintf(val, "%f ", arr[x * c + y]);
            s += val;
        }
        s += "\n";
    }
    fprintf(stdout, "%s", s.c_str());
    return;
}


void sysError(const char* string) {
    if( string )
        fprintf(stderr, "%s\n", string);
    exit(EXIT_FAILURE);
}

void matrixMul( size_t c, float* m1, float* m2, float* result ) {
    for( size_t x = 0; x < c; ++x )
    {
        for( size_t y = 0; y < c; ++y )
        {
            float sum = 0;
            for( size_t i = 0; i < c; ++i )
            {
                sum += (m1[x * c + i] * m2[i * c + y]);
            }
            result[x * c + y] = sum;
        }
    }
    return;
}



int main( int argc, char** argv ) {
    //cudaSetDevice(1);

    srand(time(NULL));
    int rows, cols;
    float* m1 = NULL;
    float* m2 = NULL;
    float* result = NULL;
    float* result2 = NULL;
    rows = cols = 1<<10; //Make this a 1024 by 1024 array

    getXYArray(m1, rows, cols);
    fillArray(m1, rows, cols);

    getXYArray(m2, rows, cols);
    fillArray(m2, rows, cols);

    getXYArray(result, rows, cols);
    getXYArray(result2, rows, cols);


    //matrixMul( cols, m1, m2, result);

    matrixMulCuda(m1, m2, result2, rows);

    
    return EXIT_SUCCESS;
}
