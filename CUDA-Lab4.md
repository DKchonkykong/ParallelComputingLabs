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

Image

<img width="498" height="528" alt="image" src="https://github.com/user-attachments/assets/fb7b739f-f220-475f-96a3-c111f258e247" />

```

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

In this exerscise it was more so configuring the CUDA sample we got and OpenGL so it can display the greyscale image you see in the output here. It was a bit confusing since I had to install libraries and I wasn't exactly sure where the DLLs should be installed but I managed to get used to where it is. 

The way that the PGM lena_bw.img loads is taht it creates a Pixel Buffer Object(PBO)
Through that the CUDA kernel writes said pixel values to the buffer and then OpenGL displays it on the screen.
It is quite interesting how this is done since it does remind me of how OpenGL works for drawing images in monogame but instead we are using CUDA plus OpenGL to achieve this. 


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
// Convert to colour
d_output[i] = make_uchar4(c* 0xff, c, c, 0);
```
This basically is similar to graphics programming for instead it's using CUDA and OpenGL. So it is turning the monochrome image into a specific colour e.g., blue.




### REFLECTION
In exercise two it made me understand how texture memory can be used to read different types of data for the image. since you can change ```make_uchar4(c*255,0,0,255)``` and it changes the monochrome image into different colours like red, greed, blue, etc.
This is quite similar to shader programming in places such as MonoGame or Unity but this is done through the use of CUDA compute kernels. 

## Exercise 3: Geometric Image Transformations


## Transform

Output

this is how it looks as is
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/282f94be-3443-4197-9470-70ac916fd3f1" />

this is how it looks when I was messing around with the keyboard controlls
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/60df21ce-cc51-4dbd-bdea-ca0b929a6d41" />



Code
```
__global__ void d_render(uchar4* d_output, uint width, uint height,
                         cudaTextureObject_t texObject, float tx, float ty)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float u = x + tx;
        float v = y + ty;

        float c = tex2D<float>(texObject, u, v);

        uchar intensity = (uchar)(c * 255.0f);
        d_output[y * width + x] = make_uchar4(intensity,intensity,intensity,255);
    }
}
```

Explanation

Translation works from modifying the tx and ty and using the WASD to transform those x and y coordinates on the image.

## Scaling

OUTPUT

This is how it looks for scale fully zooomed in and zoomed out

<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/38acc3fe-56f6-49eb-86f5-96ce315d5a4e" />



<img width="1115" height="628" alt="Screenshot 2026-03-03 134547" src="https://github.com/user-attachments/assets/ac83f2f9-e7e1-4840-986f-bc7216b6fbb0" />



CODE
```
float u = (x - cx) * scale + cx + tx;
float v = (y - cy) * scale + cy + ty;
```


Explanation

This is similar to how we did for transform except instead of moving the image around we are just scaling it up and down relative to it's center.


Rotate

Output

<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/4de20d6b-f92f-4a36-8496-925be7fc2199" />


Code
```
float xShift = x - cx;
float yShift = y - cy;

float u = cos(angle) * xShift - sin(angle) * yShift + cx;
float v = sin(angle) * xShift + cos(angle) * yShift + cy;
```


Explanation

The way to rotate an image is doine through a 2D rotation matrix and we rotate it around the center of the image. The coordinates are shifted first to the orgin and then shifted back.

## Combined Transformations

### Position by orgin
### Scale by orgin
### Rotate by orgin 

You can utilize these different transformations to perform translation, scaling and rotation at the same time as seen here:

OUTPUT

<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/a550e73e-d9e6-4d22-9c2e-689f98003887" />

This is how it looks scaling something by location and then translate it by vectors

<img width="1426" height="631" alt="image" src="https://github.com/user-attachments/assets/c3817881-4fb3-4793-940d-0ee26f24ff61" />



### REFLECTION

All in all this lab taught me how to transform an image through the use of CUDA and OpenGL and how that can be mapped not to the pixels directly but instead the kernel changes where each thread samples the texture.

Translation shifts the image by adding things like offsets to the sampling coordinates meanwhile scaling and rotation require transforming the coordinates relative to the image center.

The way it was done here is also similar to other graphics pipelines and shaders such as in OpenGL through using MonoGame or through the HLSL shaders for Unity. 



Exercise 4: Image smoothing (optional)


### REFLECTION

N/A



## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 3](CUDA-Lab3.md)
- [Cuda lab 5](CUDA-Lab5.md)

