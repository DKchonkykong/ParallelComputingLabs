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
Result matrix of C:
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
```
### Code
```
```
### Reflection


Exercise 4: Performance Comparison (Optional)

### Output

### Code

### Reflection


## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 4](CUDA-Lab4.md)
- [Cuda lab 6](CUDA-Lab6.md)

