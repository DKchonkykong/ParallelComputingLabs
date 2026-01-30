
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
// This sort of reminds me of graphics programming with how the shaders work with memory etc. I do think I need to look more into this though
// It does have parallellism and stuff it seems to be fully working now it's quite interesting in how it works do need to read more into this since i'm still new to cuda 
int main()
{
    // The actual array size doesn't seem to change what is outputed mainly cuz the prit is hard coded
    // Anyway this just seems to be setting up the arrays and pointers for the gpu
    const int arraySize = 1024;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    //Declared pointers here
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    // allocated memory on the GPU for each array using said cudaMalloc method and then it should be copied over 
    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
	cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
	cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

    //Now I have copied the input vectors both a and b from host to device then I would need to have it so it computes 
    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    
    // 3. DEVICE: Compute
    // Ok so now it should work when adding the input vectors and printing to kernel added in the size variable thing 
	int size = 5;
    printf("Launching Kernel...\n");
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);


    cudaDeviceSynchronize();
    // 4. DEVICE -> HOST: Copy Back
    //had a line that was hostToDevice here which didn't make sense and got it on the wrong way around
    //it should be working now since this is the correct way
     cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    
     //this should already clean up the memory? at least when I saw it from the helper functon
     //although it seems if this is decomented it displays the ms but if not it doesn't but apparently this is how it's supposed to work so it's ok
    //decomented for testing
    /* cudaFree(dev_c);
     cudaFree(dev_a);
     cudaFree(dev_b);*/
    // Verify Result
    //confirmed that it is working at the moment
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    //this only seems to work when I don't have the memory freed
    
   //anyway this is for the performance timing thing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);


    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(100);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element. (the error is related to intellisense being dumb so it should compile need to ignore)
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
