# CUDA Lab 6 - Procedural Textures

[Cuda lab 5](CUDA-Lab5.md) | [Next: Cuda lab 7  →](CUDA-Lab7.md)


## SMART Objectives

Procedural Image Generation: Understand how to generate images mathematically on
the GPU without loading external textures.
● Visualize Fractals: Implement the iterative algorithms for Mandelbrot and Julia sets
using parallel computing.
● Coordinate Systems: Master the transformation between Screen Space (pixels) and
World Space (mathematical units).


## Completed Tasks

Exercise 1: Setting up the Virtual Canvas

Ok so this is how it looks with the canvas working.
For whatever reason it didn't let me run it on other computers even despite that I was working on the lab 4 and tried to redo it again. it kept saying that certiain files couldn't be acesses e.g., ```#include <helper_gl.h>``` or ```#include <GL/freeglut.h>``` and I am unsure why it does this.

original version:

![alt text](<Screenshot 2026-03-13 115022.png>)

green version:

![alt text](<Screenshot 2026-03-13 115208.png>)

```
make_uchar(0,0, 0xff, 0); //red
```

Reflection:
It is similar to last week lab but instead we are just drawing colours and patterns instead of just an image. It is similar to how OpenGL does it but with the use of CUDA. It just took me longer to set up due to technical difficulties but once I had the lab 4 sheet working correctly it was easier to do task 1.

Exercise 2: Drawing a checkboard in CUDA

Output

![alt text](<Screenshot 2026-03-13 115629.png>)

Code
```
__global__ void d_render(uchar4* d_output, uint width, uint height) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;



    if ((x < width) && (y < height)) {
        const uint tileSize = 32;
        const uint checker = ((x / tileSize) + (y / tileSize)) & 1;

        if (checker == 0) {
            d_output[i] = make_uchar4(0, 0, 0, 0xff);       // black
        }
        else {
            d_output[i] = make_uchar4(0, 0, 0xff, 0); // red
        }
    }
}
```


Reflection

I now editied it so instead it has a checkerboard pattern. The major difference is taht we are using a tile size for two different ```make_uchar4``` one being black and the other being red which then represents our checkerboard pattern. This is quite similar to how other programs in openGL draw images or objects although it is usually done in c# through the vec keyword.

Exercise 3: Drawing a Circle

OUTPUT original version

![alt text](<Screenshot 2026-03-13 115929.png>)



draws a circle code normal version
```
__global__ void d_render(uchar4* d_output, uint width, uint height) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    if ((x < width) && (y < height)) {
        const float cx = 0.5f * (float)width;
        const float cy = 0.5f * (float)height;
        const float radius = 0.35f * (float)((width < height) ? width : height);

        const float dx = (float)x - cx;
        const float dy = (float)y - cy;
        const float dist2 = dx * dx + dy * dy;

        if (dist2 <= radius * radius) {
            d_output[i] = make_uchar4(0, 0xff, 0, 0xff); // green (BGRA)
        } else {
            d_output[i] = make_uchar4(0, 0, 0, 0xff);   // black background
        }
    }
}
```

this is now in mathematical space

Output

small vs full screen

small

![alt text](<Screenshot 2026-03-13 120100.png>)


full screen

![alt text](<Screenshot 2026-03-13 120117.png>)



Code
```

```


Reflection

At first I ran the circle code and it didn't look too good since I was doing it throguh screenspace but after now doing it in mathematical space it looked a lot better. The way that circle was drawn was trhough calculating the radius and then drawing a green circle for example this is quite similar to how to draw other 2D primitives in c# in openGL. 



Exercise 4: Drawing the Mandelbrot and Julia Sets

Output Mandelbrot vs julia

Mandelbrot small vs full screen

![alt text](image-2.png)


![alt text](image-1.png)

julia small vs full screen

![alt text](<Screenshot 2026-03-13 120516.png>)

![alt text](image.png)


code

```
__global__ void d_render(uchar4* d_output, uint width, uint height) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    if ((x < width) && (y < height)) {
        float u = x / (float)width;
        float v = y / (float)height;

        u = 2.0f * u - 1.0f;
        v = -(2.0f * v - 1.0f);

        u *= width / (float)height;
        u *= 4.0f;
        v *= 4.0f;

        // Mandelbrot / Julia switch
        const bool useJulia = false;
        float2 c = useJulia ? make_float2(0.25f, 0.5f) : make_float2(u, v);
        float2 z = useJulia ? make_float2(u, v) : make_float2(0.0f, 0.0f);

        const int maxIter = 200;
        int iter = 0;

        while (iter < maxIter) {
            float x2 = z.x * z.x - z.y * z.y + c.x;
            float y2 = 2.0f * z.x * z.y + c.y;
            z.x = x2;
            z.y = y2;

            if (z.x * z.x + z.y * z.y > 4.0f) {
                break;
            }
            ++iter;
        }

        if (iter == maxIter) {
            uchar shade = (uchar)(255.0f * iter / maxIter);
            d_output[i] = make_uchar4(shade, shade, 0xff, 0xff); // simple coloring
           
        }
        else {
            d_output[i] = make_uchar4(0, 0, 0, 0xff); // inside set (black)
        }
    }
}
```

Reflection

This is a bit different since I am instead calculating a mandelbrot or julia and it is a bit different since we have to calculate different x and y positions. This is quite similar to OpenGL but a bit different.


## Reflection

Overall, despite the technical issues that I had at the start of the lab I didn't find this lab all to bad and I now know how to use cuda for OpenGL more.


## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 5](CUDA-Lab5.md)
- [Cuda lab 7](CUDA-Lab7.md)

