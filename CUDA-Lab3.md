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

this is the code i will be using as the dotproduct this is a base line for when i need to use it later
```
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


Exercise 1.2: CPU+GPU Solution in CUDA
I am now going to make use of it for the GPU so now it actually does something 

```
__global__ void multiply(int* c, int* a, int* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];
}
```


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

Code bits I basically changed it to use this instead of cudaMalloc from before and use the values from the dotProduct


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

Basically now I have declared values and I don't need to use cudaMalloc since i am using static memory instead.

Exercise 3: Vector Dot-Product using Shared Memory

OUTPUT
```
=== Configuration 1: <<<1, 8>>> ===
Subtotals: 2040
Final dot product: 2040

=== Configuration 2: <<<2, 4>>> ===
Subtotals: 300 1740
Final dot product: 2040

=== Configuration 3: <<<4, 2>>> ===
Subtotals: 50 250 610 1130
Final dot product: 2040
```

CODE

```

```




REFLECTION



## Reflection

Not here sadly not yet anyway...

## Beyond the Lab (Optional)


**Navigation:**
- [Main README](README.md)
- [Main Quest 2](main-quest-2.md)

