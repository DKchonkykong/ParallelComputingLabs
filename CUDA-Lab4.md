# CUDA LAB 4

[← Cuda lab 3](CUDA-Lab3.md) | [Next: Cuda lab 5→](CUDA-Lab5.md)


## SMART Objectives

1. Master CUDA-OpenGL Interoperability: Learn how to configure a project that shares data 
between CUDA (Compute) and OpenGL (Graphics) to visualize results in real-time.
2. Understand Texture Memory: Learn to use CUDA Texture Objects to read image data 
efficiently using the GPU's read-only cache and hardware interpolation.
3. Implement Image Processing Kernels: Write kernels to manipulate pixel data.
4. Apply Geometric Transformations: Implement Translation, Scaling, and Rotation using 
coordinate mapping.
5. Understand Spatial Filtering: Implement image smoothing using neighbour-pixel averaging 
(Stencil operations).



## Completed Tasks

## Exercise 1: Project Setup & Configuration


Output
```
Starting Original Texture
GPU Device 0: "Ampere" with compute capability 8.6

CUDA device [NVIDIA GeForce RTX 3070] has 46 Multi-Processors
sdkFindFilePath <lena_bw.pgm> in ./
sdkFindFilePath <lena_bw.pgm> in ./data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/0_Introduction/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/0_Introduction/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/0_Introduction/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/1_Utilities/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/1_Utilities/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/1_Utilities/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/2_Concepts_and_Techniques/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/2_Concepts_and_Techniques/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/2_Concepts_and_Techniques/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/3_CUDA_Features/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/3_CUDA_Features/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/3_CUDA_Features/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/4_CUDA_Libraries/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/4_CUDA_Libraries/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/4_CUDA_Libraries/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/5_Domain_Specific/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/5_Domain_Specific/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/5_Domain_Specific/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/6_Performance/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/6_Performance/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/6_Performance/CudaLab4-ImageProc/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/0_Introduction/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/0_Introduction/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/0_Introduction/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/1_Utilities/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/1_Utilities/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/1_Utilities/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/2_Concepts_and_Techniques/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/2_Concepts_and_Techniques/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/2_Concepts_and_Techniques/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/3_CUDA_Features/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/3_CUDA_Features/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/3_CUDA_Features/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/4_CUDA_Libraries/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/4_CUDA_Libraries/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/4_CUDA_Libraries/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/5_Domain_Specific/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/5_Domain_Specific/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/5_Domain_Specific/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Samples/6_Performance/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Samples/6_Performance/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../Samples/6_Performance/CudaLab4-ImageProc/data/
sdkFindFilePath <lena_bw.pgm> in ../../../../Common/data/
sdkFindFilePath <lena_bw.pgm> in ../../../Common/data/
sdkFindFilePath <lena_bw.pgm> in ../../Common/data/

error: sdkFindFilePath: file <lena_bw.pgm> not found!
bicubicTexture loadImageData() could not find <lena_bw.pgm>
Exiting...

C:\Users\806395\source\repos\CudaLab4-ImageProc\x64\Debug\CudaLab4-ImageProc.exe (process 23132) exited with code 1 (0x1).
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .

```


### REFLECTION





## Exercise 2: Understanding Texture Memory & Pixels



### REFLECTION


## Exercise 3: Geometric Image Transformations


### REFLECTION

Exercise 4: Image smoothing (optional)


### REFLECTION






## Reflection



## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 3](CUDA-Lab3.md)
- [Cuda lab 5](CUDA-Lab5.md)

