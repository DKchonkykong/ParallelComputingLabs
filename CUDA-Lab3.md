# CUDA Lab 3: Memory Management & Dot Product

[← Back to Main README](README.md) | [Next: CUDA Lab 4 →](CUDA-Lab4.md)


## SMART Objectives

Establish a Baseline: Implement a vector dot product on the CPU to understand the 
logic.
2. Master Unified Memory: Learn to use cudaMallocManaged to simplify data exchange.
3. Understand Shared Memory: Learn how to allocate and use on-chip Shared Memory to 
reduce global memory latency.
4. Implement Block Reduction: Learn exactly how to map global data to shared memory, 
synchronize threads, and calculate subtotals on the GPU.

## Completed Tasks

Exercise 1: CPU-Only Solution in C++ (Baseline)


Exercise 1.1 CPU‑Only Implementation

CODE

```
int i = blockIdx.x * blockDim.x + threadIdx.x;

int dotProduct(int*a, int*b, int n)
{
	int sum = 0;
    for (int i = 0; i < n; i++) 
    {
		sum += a[i] * b[i];
    }
    return sum;
}
```
REFLECTION

In this exercise I have implemented the dot product however it's on the CPU and it is a for loop.
This is more so a baseline for the other exersices I want to use. This makes it easier to check if the later tasks are correct or not and it helps clarify the structure of what I want to do before adding stuff like parallism and having the GPU use it.


Exercise 1.2: CPU+GPU Solution in CUDA

CODE

```
__global__ void multiply(int* c, int* a, int* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];
}
```


REFLECTION
In this exercise I am now using the dot product and the GPU is processing it through the threads and blocks used meaning through this that I can multiply it and process each product e.g., ```c[i] = a[i] * b[i];```. Although since you are coping the entire vector result back to the gpu it is rather inefficient and not great for larger data sets.




Exercise 2: Vector Dot-Product using Unified Memory





Exercise 2.1: Dynamic Managed Memory

Output

```
Element-wise multiplication: {10,40,90,160,250}
Dot product (sum): 550
```


```
    cudaStatus = cudaMallocManaged(&c, size * sizeof(int));
    cudaStatus = cudaMallocManaged(&b size * sizeof(int));
    cudaStatus = cudaMallocManaged(&a, size * sizeof(int));

```
REFLECTION

In this exercise I have decided to not use cudaMalloc and cuda memcpy but instead used Cuda malloc managed. What this does is that it uses unified memory making the CPU and GPU be able to acess the same memory pointer which simplifies the code by not having the whole hosting memory transfer bits. 
After launching the kernel I made sure the GPU was synchronized by using the cuda syncronise method for it making sure the GPU finished executing correctly.

Exercise 2.2: Static Managed Memory 


Output
```
Element-wise multiplication: {10,40,90,160,250}
Dot product (sum): 550
```

Code for it
```
__device__ __managed__ int a[5];
__device__ __managed__ int b[5];
__device__ __managed__ int c[5];
```
REFLECTION

In this exercise I have made it so you don't need the need to dynamic allocation, and instead have the arrays be able to be acessed by the host and the device respectavily and automatically.

This simplified the memory model for cuda and it's good for smaller tasks to be processed.



Exercise 3: Vector Dot-Product using Shared Memory

OUTPUT


CODE




REFLECTION
Wasn't able to do this since I didn't have enough time to finish it 


## Reflection

Overall this lab taught me how to make the dot product for the CPU and how to change the cuda malloc to instead use static memory.
And having a hybrid of both CPU and GPU be used in this.
Although I wasn't able to finish the last exercise due to a lack of time sadly. 

## Beyond the Lab (Optional)


**Navigation:**
- [Main README](README.md)
- [Main Quest 2](main-quest-2.md)

