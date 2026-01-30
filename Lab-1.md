# CUDA Lab 1: Getting Started with CUDA 


[← Back to Main README](README.md) | [Next: Lab 2 →](Lab-2.md)


## SMART Objectives
Set up a CUDA project in Visual Studio 2022 using the CUDA 12.9 Runtime.
2. Analyze the Host-Device Programming Model: Understand the distinction between 
CPU (Host) and GPU (Device) execution.
3. Master Memory Management: Learn how to allocate GPU memory (cudaMalloc) and 
transfer data (cudaMemcpy).
4. Understand Kernel Execution: Learn how to launch kernels and map threads to data 
using threadIdx.x.
5. Implement Error Handling: Understand the standard pattern for catching CUDA errors.

## Completed Tasks

<img width="1115" height="628" alt="Screenshot 2026-01-30 094010" src="https://github.com/user-attachments/assets/9d979e31-4f85-44fa-9826-b1b16dcb01d6" />
This is what it looked like originally without the code alterations and with the helpers. 

<img width="1115" height="628" alt="Screenshot 2026-01-30 094029" src="https://github.com/user-attachments/assets/b9f57e89-3099-468f-b606-600132547f65" />
This is what happens when it does the error checking for when the cudaSetDevice(0) is instead 100


<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/393eddd8-803b-41b3-8312-a1a19950d48f" />

This is after I have refactored the code to remove the helpers. This is with the array size of 5.

<img width="1115" height="628" alt="Screenshot 2026-01-30 110700" src="https://github.com/user-attachments/assets/3571f075-1396-4f74-8a81-4d36f7dfb467" />

This is with an array size greater than 5 that being 1024. 
## Reflection

I have now done the introduction to CUDA. At first I didn't fully understand what exactly parallel computing was and what the difference between CPU and GPU is. However, now I understand what what CUDA is at least and how to use it and in this scenario is for calculating large arrays of numbers via parallelism. I had a few issues when trying to copy back the C array for step 4 chaning device to instead be hosting but in the end I managed to fix it and have it output correctly to the console.

Overall, I sort of now understand what CUDA is and how it can be used but I do feel like I need to get better aquainted with it in the future and understand more of what everything is doing.



**Navigation:**
- [Main README](README.md)
- [Main Quest 2](Lab-2.md)

