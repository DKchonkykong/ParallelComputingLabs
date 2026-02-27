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
img

<img width="498" height="528" alt="image" src="https://github.com/user-attachments/assets/fb7b739f-f220-475f-96a3-c111f258e247" />


Starting Original Texture
GPU Device 0: "Ampere" with compute capability 8.6

CUDA device [NVIDIA GeForce RTX 3070] has 46 Multi-Processors
sdkFindFilePath <lena_bw.pgm> in ./
sdkFindFilePath <lena_bw.pgm> in ./data/
Loaded 'lena_bw.pgm', 512 x 512 pixels

        Controls
        =/- : Zoom in/out
        [esc] - Quit


```


### REFLECTION

It was a bit hard to sort out what I needed to do for the files to work it was partly because I didn't fully understand it.
More info here.


## Exercise 2: Understanding Texture Memory & Pixels



Output

Blue
<img width="498" height="528" alt="image" src="https://github.com/user-attachments/assets/bc1612a1-abd8-4cd4-a881-a0d3a127b013" />

Red
<img width="498" height="528" alt="image" src="https://github.com/user-attachments/assets/fafb077f-8cf2-4147-8665-2f3110cca227" />


Green
<img width="498" height="528" alt="image" src="https://github.com/user-attachments/assets/4a3b6ab3-32cf-4836-bd9a-6240023bc440" />



Code
```
// Read from texture at (u,v). Returns a float 0.0 to 1.0
float c = tex2D<float>(texObj, u, v)
d_output[i] = make_uchar4(c* 0xff, c, c, 0);
```
This basically is similar to graphics programming for instead it's using CUDA and OpenGL. So it is turning the monochrome image into a specific colour e.g., blue.




### REFLECTION
The image is now blue 

## Exercise 3: Geometric Image Transformations


### REFLECTION

Exercise 4: Image smoothing (optional)


### REFLECTION






## Reflection



## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 3](CUDA-Lab3.md)
- [Cuda lab 5](CUDA-Lab5.md)

