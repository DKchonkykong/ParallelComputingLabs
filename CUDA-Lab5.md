# CUDA Lab 5

[Cuda lab 4](CUDA-Lab4.md) | [Next: Cuda lab 6  →](CUDA-Lab6.md)


## SMART Objectives
* Understand Matrix Multiplication: Master the rules of performing row-by-column
multiplication.
● Implement Linear Algebra on GPU: Learn how to map 2D Matrix data to CUDA threads.
● Single Block vs. Multi-Block: Implement two versions of the algorithm to understand
scalability limits.
● Performance Analysis: Compare the execution time of CPU vs. GPU implementations
for large matrices.

## Completed Tasks

Exercise 1: CPU Implementation (Baseline)


### Output

So when I first ran it I got this error and these values ```Run-Time Check Failure #2 - Stack around the variable C was corrupted````

```
Result matrix of C:
-751619200.0 -751619200.0 -751619200.0
-1610612480.0 -1610612480.0 -1610612480.0
-2469605888.0    0.0    0.0

```

### Code

Matrix multiplication code for it
```
void MatrixMultiplyCPU(float* A, float* B, float* C, int widthA, int heightA, int widthB, int heightB);

//widths stuff
int widthA = 4;
int heightA = 3;
int widthB = 3;
int heightB = 2;
//matrices
float A[12]
{ 1, 2, 3, 4, 5 ,6, 7, 8, 9,10, 11 ,12 };
float B[8]
{ 1,2,3,4,5,6,7,8 };
float C[6]
{ 0,1,2,3,4 };

//calls the matrix multiplication method to calculate it all
//currently not working atm
MatrixMultiplyCPU(A, B, C, widthA, heightA, widthB, heightB);

//prints result of matrix multiplication
printf("Result matrix of C:\n");
for (int i = 0; i < heightA; i++)
{
    for (int j = 0; j < widthB; j++)
    {
        printf("%6.1f ", C[i * widthB + j]);
    }
    printf("\n");
}
return 0;

void MatrixMultiplyCPU(float* A, float* B, float* C, int widthA, int heightA, int widthB, int heightB)
{
    for (int row = 0; row < heightA; row++)
    {
        for (int col = 0; col < widthB; col++)
        {
            float sum = 0;
            for (int k = 0; k < widthA; k++)
            {
                sum += A[row * widthA + k] * B[k * widthA + col];
            }
            C[row * widthB + col] = sum;
        }
    }

}

```

Ended up fixing it by this
Output

```
Result matrix of C:
  50.0   60.0
 114.0  140.0
 178.0  220.0
```

Code

```
int main()
{
    //widths stuff
    int widthA = 4;
    int heightA = 3;
    int widthB = 2;  // Changed from 3 to 2
    int heightB = 4;  // B must have 4 rows to match A's 4 columns
    //matrices
    float A[12]
    { 1, 2, 3, 4, 5 ,6, 7, 8, 9,10, 11 ,12 };
    float B[8]
    { 1,2,3,4,5,6,7,8 };
    float C[6] = { 0 };  // Fixed: proper initialization with all zeros

    //calls the matrix multiplication method to calculate it all
    MatrixMultiplyCPU(A, B, C, widthA, heightA, widthB, heightB);

    //prints result of matrix multiplication
    printf("Result matrix of C:\n");
    for (int i = 0; i < heightA; i++)
    {
        for (int j = 0; j < widthB; j++)
        {
            printf("%6.1f ", C[i * widthB + j]);
        }
        printf("\n");
    }
    return 0;
}

//CPU matrix code excersice 1 epic

void MatrixMultiplyCPU(float* A, float* B, float* C, int widthA, int heightA, int widthB, int heightB)
{
    for (int row = 0; row < heightA; row++)
    {
        for (int col = 0; col < widthB; col++)
        {
            float sum = 0;
            for (int k = 0; k < widthA; k++)
            {
                sum += A[row * widthA + k] * B[k * widthB + col];  // Fixed: widthB not widthA
            }
            C[row * widthB + col] = sum;
        }
    }
}

```




### Reflection

It now works I wasn't sure what I was doing oops.

Exercise 2: Single-Block CUDA Implementation

### Output
```
Result matrix of C (CPU):
  50.0   60.0
 114.0  140.0
 178.0  220.0

Result of matrix C from GPU:
  50.0   60.0
 114.0  140.0
 178.0  220.0
```
### Code
```
//pretty similar to the task 1 matrix multiplication buut this is with the GPU 
__global__ void MatrixMultiplyKernel(float * A, float * B, float * C, int widthA, int widthB)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    float sum = 0;
        for (int k = 0; k < widthA; k++)
        {
            sum += A[row * widthA + k] * B[k * widthB + col];
        }
        C[row * widthB + col] = sum;
}

int main()
{

    //widths stuff
    int widthA = 4;
    int heightA = 3;
    int widthB = 2;  // it changed from 3 to 2
    int heightB = 4;  // B must have 4 rows to match A's 4 columns
    
    //matricies stuff here
    float A[12]
    { 1, 2, 3, 4, 5 ,6, 7, 8, 9,10, 11 ,12 };
    float B[8]
    { 1,2,3,4,5,6,7,8 };
    float C[6] = { 0 };  // it is now fixed it has proper initialization with all zeros



    //exerscise 2 here!!!
    //allocating memory for the GPU here
    float* dev_A = 0;
    float* dev_B = 0;
    float* dev_C = 0;


    cudaMalloc((void**)&dev_A, heightA * widthA * sizeof(float));
    cudaMalloc((void**)&dev_B, heightB * widthB * sizeof(float));
    cudaMalloc((void**)&dev_B, heightA * widthB * sizeof(float));

    //copying data to the GPU
    cudaMemcpy(dev_A, A, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, heightB * widthB * sizeof(float), cudaMemcpyHostToDevice);

    //this just seems for the C stuff?
    dim3 dimBlock(widthB, heightA);

	MatrixMultiplyKernel <<< 1, dimBlock >>> (dev_A, dev_B, dev_C, widthA, widthB);

    cudaDeviceSynchronize();
	
	cudaMemcpy(C, dev_C, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result of matrix C from GPU:\n");
	for (int i = 0; i < heightA; i++)
        {
        for (int j = 0; j < widthB; j++)
        {
            printf("%6.1f ", C[i * widthB + j]);
        }
        printf("\n");
    }
    //freeing memory on the GPU
    cudaFree(dev_A);
    cudaFree(dev_B);
	cudaFree(dev_C);

    return 0;
}
```

### Reflection

Did this it worky it is different to task 1 where I had to do it via cpu now we doin on the GPU similar to other execises epic 

Exercise 3: Multi-Block CUDA Implementation (2D
Grid)

### Output
```
esult matrix of C (CPU):
  50.0   60.0
 114.0  140.0
 178.0  220.0

Result of matrix C from GPU (Single Block):
  50.0   60.0
 114.0  140.0
 178.0  220.0

 Exercise 3 Large matrix test with grid of blocks
Grid dimensions: 32 x 32 blocks
Block dimensions: 16 x 16 threads
Total threads: 262144

Sample of result (first 5x5 elements):
  1024.0   1024.0   1024.0   1024.0   1024.0
  1024.0   1024.0   1024.0   1024.0   1024.0
  1024.0   1024.0   1024.0   1024.0   1024.0
  1024.0   1024.0   1024.0   1024.0   1024.0
  1024.0   1024.0   1024.0   1024.0   1024.0

Expected value per element: 1024.0

```
### Code
```
    // this is for exercise 3 we are using grid of blocks this is the main code for it 
    printf("\n Exercise 3 Large matrix test with grid of blocks \n");

    int largeWidthA = 512;
    int largeHeightA = 512;
    int largeWidthB = 512;
    int largeHeightB = 512;

    // Allocate host memory
    float* largeA = new float[largeHeightA * largeWidthA];
    float* largeB = new float[largeHeightB * largeWidthB];
    float* largeC = new float[largeHeightA * largeWidthB];

    // Initialize matrices
    for (int i = 0; i < largeHeightA * largeWidthA; i++)
        largeA[i] = 1.0f;  // Simple initialization
    for (int i = 0; i < largeHeightB * largeWidthB; i++)
        largeB[i] = 2.0f;

    // Allocate device memory
    float* dev_largeA = 0;
    float* dev_largeB = 0;
    float* dev_largeC = 0;

    cudaMalloc((void**)&dev_largeA, largeHeightA * largeWidthA * sizeof(float));
    cudaMalloc((void**)&dev_largeB, largeHeightB * largeWidthB * sizeof(float));
    cudaMalloc((void**)&dev_largeC, largeHeightA * largeWidthB * sizeof(float));

    // Copy to device
    cudaMemcpy(dev_largeA, largeA, largeHeightA * largeWidthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_largeB, largeB, largeHeightB * largeWidthB * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    // Each block is TILE_SIZE x TILE_SIZE (16x16)
    // Grid needs enough blocks to cover the entire matrix
    dim3 dimBlockLarge(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((largeWidthB + TILE_SIZE - 1) / TILE_SIZE,  // Ceiling division
        (largeHeightA + TILE_SIZE - 1) / TILE_SIZE);

    printf("Grid dimensions: %d x %d blocks\n", dimGrid.x, dimGrid.y);
    printf("Block dimensions: %d x %d threads\n", dimBlockLarge.x, dimBlockLarge.y);
    printf("Total threads: %d\n", dimGrid.x * dimGrid.y * dimBlockLarge.x * dimBlockLarge.y);

    // Launch kernel with grid of blocks
    MatrixMultiplyKernelGrid << < dimGrid, dimBlockLarge >> > (dev_largeA, dev_largeB, dev_largeC,
        largeWidthA, largeWidthB, largeHeightA);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(largeC, dev_largeC, largeHeightA * largeWidthB * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result (print a small sample)
    printf("\nSample of result (first 5x5 elements):\n");
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            printf("%8.1f ", largeC[i * largeWidthB + j]);
        }
        printf("\n");
    }

    // Expected result: each element should be 512 * 1.0 * 2.0 = 1024.0
    printf("\nExpected value per element: %.1f\n", (float)largeWidthA * 1.0f * 2.0f);

    // Cleanup
    cudaFree(dev_largeA);
    cudaFree(dev_largeB);
    cudaFree(dev_largeC);
    

    return 0;
}

```
### Reflection

From what I understand this is similar to exercise two ut instead we are using more than one block. 

Exercise 4: Performance Comparison (Optional)

### Output

```
Performance comparison GPU vs CPU for 1024x1024 matricies

Calculating on CPU...
CPU Time: 3.0580 seconds

Calculating on GPU...
GPU Time: 0.0147 seconds (14.72 ms)

 Performace comparison GPU vs CPU
Matrix Size: 1024 x 1024
CPU Time: 3.0580 seconds
GPU Time: 0.0147 seconds
Speedup: 207.80x faster on GPU
Performance Gain: 99.52%

 Verifying if everything is correct
CPU[0][0]=203.67, GPU[0][0]=203.67, Diff=0.000000
CPU[0][1]=199.61, GPU[0][1]=199.61, Diff=0.000000
CPU[0][2]=195.65, GPU[0][2]=195.65, Diff=0.000000
CPU[1][0]=207.86, GPU[1][0]=207.86, Diff=0.000000
CPU[1][1]=205.49, GPU[1][1]=205.49, Diff=0.000000
CPU[1][2]=198.93, GPU[1][2]=198.93, Diff=0.000000
CPU[2][0]=202.40, GPU[2][0]=202.40, Diff=0.000000
CPU[2][1]=200.96, GPU[2][1]=200.96, Diff=0.000000
CPU[2][2]=199.79, GPU[2][2]=199.79, Diff=0.000000

Results match: YES
```
### Code

```
    // Exercise 4: Performance comparison CPU vs GPU for 1024x1024 matrices
    printf("\nPerformance comparison GPU vs CPU for 1024x1024 matricies\n");

    int size1024 = 1024;
    int widthA_1024 = size1024;
    int heightA_1024 = size1024;
    int widthB_1024 = size1024;
    int heightB_1024 = size1024;

    // Allocate host memory for 1024x1024 matrices
    float* A_1024 = new float[heightA_1024 * widthA_1024];
    float* B_1024 = new float[heightB_1024 * widthB_1024];
    float* C_CPU_1024 = new float[heightA_1024 * widthB_1024];
    float* C_GPU_1024 = new float[heightA_1024 * widthB_1024];

    // Initialize matrices with random values
    for (int i = 0; i < heightA_1024 * widthA_1024; i++)
        A_1024[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < heightB_1024 * widthB_1024; i++)
        B_1024[i] = (float)(rand() % 10) / 10.0f;

    // CPU Timing
    printf("\nCalculating on CPU...\n");
    clock_t cpu_start = clock();

    MatrixMultiplyCPU(A_1024, B_1024, C_CPU_1024, widthA_1024, heightA_1024, widthB_1024, heightB_1024);

    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU Time: %.4f seconds\n", cpu_time);

    //GPU Timing
    printf("\nCalculating on GPU...\n");

    // Allocate device memory
    float* dev_A_1024 = 0;
    float* dev_B_1024 = 0;
    float* dev_C_1024 = 0;

    cudaMalloc((void**)&dev_A_1024, heightA_1024 * widthA_1024 * sizeof(float));
    cudaMalloc((void**)&dev_B_1024, heightB_1024 * widthB_1024 * sizeof(float));
    cudaMalloc((void**)&dev_C_1024, heightA_1024 * widthB_1024 * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy data to device
    cudaMemcpy(dev_A_1024, A_1024, heightA_1024 * widthA_1024 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B_1024, B_1024, heightB_1024 * widthB_1024 * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 dimBlock_1024(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid_1024((widthB_1024 + TILE_SIZE - 1) / TILE_SIZE,
        (heightA_1024 + TILE_SIZE - 1) / TILE_SIZE);

    // Start GPU timing
    cudaEventRecord(start);

    // Launch kernel
    MatrixMultiplyKernelGrid << <dimGrid_1024, dimBlock_1024 >> > (dev_A_1024, dev_B_1024, dev_C_1024,
        widthA_1024, widthB_1024, heightA_1024);

    // Stop GPU timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    double gpu_time = gpu_time_ms / 1000.0;  // Convert to seconds

    printf("GPU Time: %.4f seconds (%.2f ms)\n", gpu_time, gpu_time_ms);

    // Copy result back
    cudaMemcpy(C_GPU_1024, dev_C_1024, heightA_1024 * widthB_1024 * sizeof(float), cudaMemcpyDeviceToHost);

	// Performance GPU vs CPU
    printf("\n Performace comparison GPU vs CPU\n");
    printf("Matrix Size: %d x %d\n", size1024, size1024);
    printf("CPU Time: %.4f seconds\n", cpu_time);
    printf("GPU Time: %.4f seconds\n", gpu_time);
    printf("Speedup: %.2fx faster on GPU\n", cpu_time / gpu_time);
    printf("Performance Gain: %.2f%%\n", ((cpu_time - gpu_time) / cpu_time) * 100.0);

    // Verify correctness - compare a few elements
    printf("\n Verifying if everything is correct\n");
    bool correct = true;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int idx = i * widthB_1024 + j;
            float diff = fabs(C_CPU_1024[idx] - C_GPU_1024[idx]);
            printf("CPU[%d][%d]=%.2f, GPU[%d][%d]=%.2f, Diff=%.6f\n",
                i, j, C_CPU_1024[idx], i, j, C_GPU_1024[idx], diff);
            if (diff > 0.01f)  // Tolerance for floating point comparison
                correct = false;
        }
    }
    printf("\nResults match: %s\n", correct ? "YES" : "NO");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_A_1024);
    cudaFree(dev_B_1024);
    cudaFree(dev_C_1024);
    delete[] A_1024;
    delete[] B_1024;
    delete[] C_CPU_1024;
    delete[] C_GPU_1024;

    return 0;
}
```

### Reflection

Tested it now it is quite interesting the difference between both of them tbh

## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 4](CUDA-Lab4.md)
- [Cuda lab 6](CUDA-Lab6.md)

