#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "gaussCuda.hh"
#include "CycleTimer.hh"

#define NO_OF_THREADS 32

#define THREADS_PER_BLOCK 32

extern float toBW(int bytes, float sec);

__global__ void
gauss_kernel_H(float *img, float *kernel, float *res, float *temp, int h, int w, int k) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >=w || y >=h)
        return;

    int center = k/2;
    float out = 0.0;
    int row = y * w;
    for(int i = -center; i <= center; i++)
    {
        int x1 = x + i;
        x1 = max(x1,0);
        x1 = min(x1,w-1);
        //if(x1 >=0 && x1 < w)
        out += temp[row + x1]*kernel[i+center];    
    }

    res[row + x] = out;
}

__global__ void
gauss_kernel_V(float *img, float *kernel, float *res, float *temp, int h, int w, int k) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >=w || y >=h)
        return;

    int center = k/2;
    float out = 0.0;
    for(int i = -center; i <= center; i++)
    {
        int y1 = y + i;
        y1 = max(y1,0);
        y1 = min(y1,h-1);
        //if(y1 >=0 && y1<h)
        out += img[y1*w + x]*kernel[i+center];    
    }   

    temp[y*w + x] = out;
}


__global__ void
gauss_kernel(float *img, float *kernel, float *res, int h, int w, int k) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >=w || y >=h)
        return;

    int center = k/2;
    float out = 0.0;
    for(int i = -center; i <= center; i++)
    {   
        int x1 = x + i;
        x1 = max(x1,0);
        x1 = min(x1,w-1);
        for(int j = -center; j <= center; j++)
        {
            int y1 = y + j;
            y1 = max(y1,0);
            y1 = min(y1,h-1);
            out += img[y1*w + x1]* kernel[(i+center)*k+(j+center)]; 
        }   
    }   

    res[y*w + x] = out;
}

/*__global__ void
gauss_kernel_tiled1(float *img, float *kernel, float *res, int h, int w, int k)
{
    int h1 = PIXELS_PER_BLOCK + k-1;
    int w1 = h1;
    int center = k/2;
    __shared__ float patch[h1][w1];

    __shared__ float temp[PIXELS_PER_BLOCK][h1-1];

    int x = threadIdx.x; 
    int y = threadIdx.y;

    int globalX = blockIdx.x * blockDim.x + x;
    int globalY = blockIdx.y * blockDim.y + y;

    patch[x+center][y+center] = img[globalX][globalY];
    if(x < center) 
    {
        int row = max(0,globalX-center+x);
        patch[x][y+center] = img[row][y];
        row = min(w-1,globalX + PIXELS_PER_BLOCK + x);
        patch[w1+x][y+center] = img[row][y];
    }

    if(y < center)
    {  
        int col = max(0,globalY-center+y);
        patch[x][y+center] = img[x][col];
        col = min(h-1,globalY + PIXELS_PER_BLOCK + y);
        patch[x][h1+y+center] = img[x][col;
    } 

    __syncthreads();

    if(x >= w || y >= h)
        return;

    //Horizontal Blur

    int x1 = x+center;
    int y1 = y+center;    

    float out = 0.0;
    for(int p = -center; p < center; p++)
        out += patch[x1+p][y1] * kernel[p+center];     

    temp[x][y] = out;    

    __syncthreads();

    //Vertical Blur

    for(int p = -center; p < center; p++)
        out += patch[x][y+p] * kernel[p+center];       

    res[globalX][globalY] = out;
}*/

__global__ void
gauss_kernel_tiled2(float *img, float *kernel, float *res, int h, int w, int k)
{
    int center = k/2;
    int h1 = THREADS_PER_BLOCK - k + 1;
    int w1 = h1;
    __shared__ float patch[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    __shared__ float temp[THREADS_PER_BLOCK][THREADS_PER_BLOCK];

    int x = threadIdx.y;
    int y = threadIdx.x;

    int globalX = blockIdx.y * w1 + x - center;
    int globalY = blockIdx.x * h1 + y - center;    

    int col = max(0,globalX);
    col = min(col,w-1);
    int row = max(0,globalY);
    row = min(row,h-1);

    patch[y][x] = img[row*w + col];

    __syncthreads();

    float out;
    if(((x >=center) && (x < center + w1)))
    {
        //Horizontal Blur
        out = 0.0;
        for(int p = -center; p <= center; p++)
            out += patch[y][x+p] * kernel[p+center];

        temp[y][x] = out;
    }

    __syncthreads();

    if(((y < center) || (y >= center + h1)) || (x < center) || (x >= center + w1))
        return;

    //Vertical Blur
    out = 0.0;
    for(int p = -center; p <= center; p++)
        out += temp[y+p][x] * kernel[p+center];

    res[globalY*w + globalX] = out;
}

void gaussCuda(const float *img, float *kernel, float *res, int h, int w, int k) 
{
    dim3 threadsPerBlock(NO_OF_THREADS,NO_OF_THREADS);
    dim3 numBlocks;
    
    /*numBlocks.x = w / threadsPerBlock.x + 1;
    numBlocks.y = h / threadsPerBlock.y + 1;
    */

    numBlocks.x = (w + threadsPerBlock.x - 1)/threadsPerBlock.x;
    numBlocks.y = (h + threadsPerBlock.y - 1)/threadsPerBlock.y;

    float *device_img;
    float *device_k;
    float *device_res;
    float *device_temp;

    int imgSize = sizeof(float)*h*w;
    int kSize = sizeof(float)*k;

    cudaMalloc(&device_img,imgSize);
    cudaMalloc(&device_k,kSize);
    cudaMalloc(&device_res,imgSize);
    cudaMalloc(&device_temp,imgSize);

    //double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(device_img, img, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, kernel, kSize, cudaMemcpyHostToDevice);

    //double kernelStart = CycleTimer::currentSeconds();

    gauss_kernel_V<<<numBlocks,threadsPerBlock>>>(device_img, device_k, device_res, device_temp, h, w, k);
    cudaDeviceSynchronize();
    gauss_kernel_H<<<numBlocks,threadsPerBlock>>>(device_img, device_k, device_res, device_temp, h, w, k);
    cudaDeviceSynchronize();

//    gauss_kernel<<<numBlocks,threadsPerBlock>>>(device_img, device_k, device_res, h, w, k);
  
    //double kernelEnd = CycleTimer::currentSeconds();

    cudaMemcpy(res, device_res, imgSize, cudaMemcpyDeviceToHost);

    //double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
/*
    double overallDuration = endTime - startTime;
    double overallKernel = kernelEnd - kernelStart;
    printf("Overall: %.3f ms ; Only GPU computation: %.3f ms\n", 1000.f * overallDuration, 1000.f * overallKernel);
*/

    cudaFree(device_img);
    cudaFree(device_k);
    cudaFree(device_res);
    cudaFree(device_temp);
}


void gaussCuda2(const float *img, float *kernel, float *res, int h, int w, int k)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
    dim3 numBlocks;
    int pixelsPerBlock = THREADS_PER_BLOCK - k + 1;
    numBlocks.y = (w + pixelsPerBlock - 1)/pixelsPerBlock;
    numBlocks.x = (h + pixelsPerBlock - 1)/pixelsPerBlock;

    float *device_img;
    float *device_k;
    float *device_res;

    int imgSize = sizeof(float)*h*w;
    int kSize = sizeof(float)*k;

    cudaMalloc(&device_img,imgSize);
    cudaMalloc(&device_k,kSize);
    cudaMalloc(&device_res,imgSize);

   // double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(device_img, img, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, kernel, kSize, cudaMemcpyHostToDevice);

   // double kernelStart = CycleTimer::currentSeconds();
    gauss_kernel_tiled2<<<numBlocks,threadsPerBlock>>>(device_img, device_k, device_res, h, w, k);
    cudaDeviceSynchronize();
   // double kernelEnd = CycleTimer::currentSeconds();
    cudaMemcpy(res, device_res, imgSize, cudaMemcpyDeviceToHost);

   // double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

   // double overallDuration = endTime - startTime;
   // double overallKernel = kernelEnd - kernelStart;
   // printf("Overall: %.3f ms ; Only GPU computation: %.3f ms\n", 1000.f * overallDuration, 1000.f * overallKernel );
//    printf("H = %d, W = %d, k = %d  ; Overall: %.3f ms\n", h,w,k,1000.f * overallDuration);


    cudaFree(device_img);
    cudaFree(device_k);
    cudaFree(device_res);
}

void gaussCuda3(const float *img, float *kernel, float *res, int h, int w, int k) 
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK,THREADS_PER_BLOCK);
    dim3 numBlocks;
    int pixelsPerBlock = THREADS_PER_BLOCK - k + 1;

    numBlocks.y = (w + pixelsPerBlock - 1)/pixelsPerBlock;
    numBlocks.x = (h + pixelsPerBlock - 1)/pixelsPerBlock;

    /*printf("w = %d, h = %d\n",w,h);
    printf("k = %d\n",k);
    printf("X Bloxks = %d, y Blocks = %d\n",numBlocks.x,numBlocks.y);
*/

    static float *device_img = NULL;
    static float *device_k = NULL;
    static float *device_res = NULL;

    static float *prevImagePtr = NULL;
    int isPrev = 0;

    int imgSize = sizeof(float)*h*w;
    int kSize = sizeof(float)*k;

    if(prevImagePtr == img)
        isPrev = 1;

    if(!isPrev)
    {
        if(device_img != NULL)
        {
            cudaFree(device_img);
            cudaFree(device_res);
        }
        //printf("Mallocing Device Memory for Image and Result\n");
        cudaMalloc(&device_img,imgSize);
        cudaMalloc(&device_res,imgSize);
    }
    
    cudaMalloc(&device_k,kSize);

    //cudaMalloc(&device_res,imgSize);

   // double startTime = CycleTimer::currentSeconds();

    if(!isPrev)
    {
        cudaMemcpy(device_img, img, imgSize, cudaMemcpyHostToDevice);
        //printf("Copying Image\n");
    }    

    cudaMemcpy(device_k, kernel, kSize, cudaMemcpyHostToDevice);

  //  double kernelStart = CycleTimer::currentSeconds();
    gauss_kernel_tiled2<<<numBlocks,threadsPerBlock>>>(device_img, device_k, device_res, h, w, k);
    cudaDeviceSynchronize();
  //  double kernelEnd = CycleTimer::currentSeconds();
    cudaMemcpy(res, device_res, imgSize, cudaMemcpyDeviceToHost);

  //  double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

/*    double overallDuration = endTime - startTime;
    double overallKernel = kernelEnd - kernelStart;
    printf("Overall: %.3f ms ; Only GPU computation: %.3f ms ; Transfer Time: %.3f ms\n", 1000.f * overallDuration, 1000.f * overallKernel, 1000.f *(overallDuration - overallKernel));
*/
    prevImagePtr = (float *)(&img[0]);    
    cudaFree(device_k);
}
