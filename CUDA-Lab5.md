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
```
### Code
```
```
### Reflection

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

