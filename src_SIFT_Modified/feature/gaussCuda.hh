#ifndef GAUSS_CUDA_H
#define GAUSS_CUDA_H

#include "CycleTimer.hh"

void gaussCuda(const float *img, float *kernel, float *res, int h, int w, int k);
void gaussCuda2(const float *img, float *kernel, float *res, int h, int w, int k);
void gaussCuda3(const float *img, float *kernel, float *res, int h, int w, int k);

#endif
