#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//this is similar for the add kernel but instead for doing the dot product stuff here

__global__ void PerElement_AtimesB(int* c, int* a, int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

// Exercise 3: Kernel using Shared Memory for per-block reduction
__global__ void dotProductSharedMem(int* c, int* a, int* b)
{
    // 1. Define shared memory (size should match blockDim.x)
    __shared__ int dataPerBlock[8]; // Max threads per block we'll use

    // 2. Identify the data - calculate global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 3. Load data into shared memory (perform multiplication here)
    dataPerBlock[threadIdx.x] = a[i] * b[i];

    // 5. Thread synchronization - wait for all threads to finish writing
    __syncthreads();

    // 4. Compute the subtotal (only thread 0 of each block)
    if (threadIdx.x == 0)
    {
        int subtotal = 0;
        for (int k = 0; k < blockDim.x; k++)
        {
            subtotal += dataPerBlock[k];
        }
        // Store the subtotal in global array 'c', indexed by Block ID
        c[blockIdx.x] = subtotal;
    }
}


//the dot product will be used for the rest of the program as like a base standard
int dotProduct(int* a, int* b, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

//declared values outside of main 
__device__ __managed__ int a[8];
__device__ __managed__ int b[8];
__device__ __managed__ int c[8];

int main()
{
    const int arraySize = 8;

    // Initialize values for 8 elements
    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
    a[4] = 5; a[5] = 6; a[6] = 7; a[7] = 8;

    b[0] = 10; b[1] = 20; b[2] = 30; b[3] = 40;
    b[4] = 50; b[5] = 60; b[6] = 70; b[7] = 80;

    // Test different configurations
    int numBlocks, threadsPerBlock;

    // Configuration 1: <<<1, 8>>> (1 Block, 8 Threads)
    printf("=== Configuration 1: <<<1, 8>>> ===\n");
    numBlocks = 1;
    threadsPerBlock = 8;
    dotProductSharedMem << <numBlocks, threadsPerBlock >> > (c, a, b);
    cudaDeviceSynchronize();

    int sum = 0;
    for (int i = 0; i < numBlocks; i++)
    {
        sum += c[i];
    }
    printf("Subtotals: ");
    for (int i = 0; i < numBlocks; i++)
    {
        printf("%d ", c[i]);
    }
    printf("\nFinal dot product: %d\n\n", sum);

    // Configuration 2: <<<2, 4>>> (2 Blocks, 4 Threads)
    printf("=== Configuration 2: <<<2, 4>>> ===\n");
    numBlocks = 2;
    threadsPerBlock = 4;
    dotProductSharedMem << <numBlocks, threadsPerBlock >> > (c, a, b);
    cudaDeviceSynchronize();

    sum = 0;
    for (int i = 0; i < numBlocks; i++)
    {
        sum += c[i];
    }
    printf("Subtotals: ");
    for (int i = 0; i < numBlocks; i++)
    {
        printf("%d ", c[i]);
    }
    printf("\nFinal dot product: %d\n\n", sum);

    // Configuration 3: <<<4, 2>>> (4 Blocks, 2 Threads)
    printf("=== Configuration 3: <<<4, 2>>> ===\n");
    numBlocks = 4;
    threadsPerBlock = 2;
    dotProductSharedMem << <numBlocks, threadsPerBlock >> > (c, a, b);
    cudaDeviceSynchronize();

    sum = 0;
    for (int i = 0; i < numBlocks; i++)
    {
        sum += c[i];
    }
    printf("Subtotals: ");
    for (int i = 0; i < numBlocks; i++)
    {
        printf("%d ", c[i]);
    }
    printf("\nFinal dot product: %d\n\n", sum);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc(&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return cudaStatus;
}