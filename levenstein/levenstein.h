#ifndef LEVENSTEIN_H
#define LEVEHSTEIN_H

#define STRIPE_MAX 512
#define WARP_MAX 32

void levensteinCuda(char* s1, char* s2, size_t* result, size_t n);

#endif