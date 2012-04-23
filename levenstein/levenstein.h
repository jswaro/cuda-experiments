#ifndef LEVENSTEIN_H
#define LEVEHSTEIN_H

#define TILE_SIZE 16
#define TILES_PER_BLOCK 2
#define MAX_BLOCKS 8
#define CMAX(a,b) (a-((a-b)&((a-b)>>31)))
#define STRIPE_MAX 512
#define WARP_MAX 32

void levensteinCuda(char* s1, char* s2, size_t* result, size_t n);

#endif
