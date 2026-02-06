# CUDA LAB 2

[← Back to Main README](README.md) | [Next: Cuda lab 3→](CUDA-Lab3.md)


## SMART Objectives
1. Master Thread Indexing: Learn how to calculate the Global Index of a thread to map it 
to a specific data element in an array.
2. Understanding Grids and Blocks: Move beyond single-block kernels to use multiple 
blocks (gridDim) and multiple threads (blockDim).
3. Multi-Dimensional Configuration: Learn to organize threads in 2D and 3D using dim3.
4. Matrix Operations: Apply 2D indexing to perform Matrix Addition.


## Completed Tasks
Task 1 - Thread Index Identification in CUDA:
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/bcc65b78-f766-447d-be66-5c995d079d31" />
This was adding blocks and threads before fixing the global index function there wasn't much of a change even when adding the additional threads/blocks
Task 2 - Global Thread Index for Multiple 1D Thread Blocks:
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/c3283e4c-21c9-4366-92fc-504c98f78b4c" />
Since the global index function is now added there is a noticable difference in terms of performance
Task 3 - 2D Thread Blocks:
Listed all the possible variations
Task 4 - Global Thread Index for 2D Thread Blocks:
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/357b4302-f251-4f64-b461-fe9e0b57d31a" />

Task 5 - Global Thread Index for Multiple 2D
Thread Blocks:
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/3a7135ce-2388-4abd-8e8c-0ddee9a31346" />
Used the vector addition example of addKernel << <dim3(2, 3), dim3(2, 2) >> > (dev_c, dev_a, dev_b);
and it now gave me back this result


Task 6 - Matrix Addition (Optional):
Didn't bother doing this at the moment might do it later who knows.


## Reflection
I think it was interesting using the multiple threads and blocks for 1D and 2D there was some differences in performance as well depending on how many were used. At first I had some issues trying to see if the changes in performance when doing the 1D threads and 2D threads were correct but later I managed to fix it and actually make it work. Overall, I sort of understand how to use threads for CUDA



## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 1](CUDA-Lab1.md)
- [Cuda lab 3](CUDA-Lab3.md)

