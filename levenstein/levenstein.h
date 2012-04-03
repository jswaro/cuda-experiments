#ifndef MATRIX_H
#define MATRIX_H

#define TILE_SIZE 16
#define TILES_PER_BLOCK 2
#define MAX_BLOCKS 8

void matrixMulCuda(float* m1, float* m2, float* &result, size_t size);

#endif