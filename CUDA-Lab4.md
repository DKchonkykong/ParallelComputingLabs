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


Transform

<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/c479d36e-7c67-433c-b8f0-cac682bf6c29" />


Code
```
extern "C" void render(int width, int height,  dim3 blockSize, dim3 gridSize,
     uchar4 * output, float tx, float ty) {

    float scale = 1, cx = 0, cy = 0;

        d_render << <gridSize, blockSize >> > (output, width, height, tx, ty, 1,
            0, 0, rgbaTexdImage);


    getLastCudaError("kernel failed");
```

it is supposed to move the image around doesn't really do that? at least for transform 

<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/39b7da4b-e064-44c6-a43d-33067be73d82" />


Managed to get it working via using the texture objects api
I think it worked because it was newer and you can manage multiple textures?

CODE
```
#include <cuda_runtime.h>
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

// Texture object
cudaTextureObject_t texObj = 0;
cudaArray* d_imageArray = 0;

// CUDA kernel using texture object
__global__ void d_render(uchar4* d_output, uint width, uint height, 
                         cudaTextureObject_t texObject, float tx, float ty) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Apply translation using float2
        float2 T = make_float2(tx, ty);
        float u = x + T.x;
        float v = y + T.y;

        // Read pixel color with translated coordinates
        float c = tex2D<float>(texObject, u, v);

        // Convert to grayscale output
        uchar intensity = (uchar)(c * 255.0f);
        d_output[y * width + x] = make_uchar4(intensity, intensity, intensity, 255);
    }
}

extern "C" void initTexture(int imageWidth, int imageHeight, uchar* h_data) {
    // Create channel descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, 
                                                               cudaChannelFormatKindUnsigned);

    // Allocate CUDA array
    checkCudaErrors(cudaMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight));

    // Copy data to array
    checkCudaErrors(cudaMemcpyToArray(d_imageArray, 0, 0, h_data, 
                                       imageWidth * imageHeight * sizeof(uchar), 
                                       cudaMemcpyHostToDevice));

    // Create resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_imageArray;

    // Create texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;  // Use pixel coordinates

    // Create texture object
    checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
}

extern "C" void freeTexture() {
    if (texObj) {
        checkCudaErrors(cudaDestroyTextureObject(texObj));
        texObj = 0;
    }
    if (d_imageArray) {
        checkCudaErrors(cudaFreeArray(d_imageArray));
        d_imageArray = 0;
    }
}

extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, 
                       uchar4* d_output, float tx, float ty) {
    // Launch kernel with translation parameters
    d_render<<<gridSize, blockSize>>>(d_output, width, height, texObj, tx, ty);
    
    // Check for errors
    getLastCudaError("kernel failed");
}
```

this is how it looks as is
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/282f94be-3443-4197-9470-70ac916fd3f1" />

this is how it looks when I was messing around with the keyboard controlls
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/60df21ce-cc51-4dbd-bdea-ca0b929a6d41" />

so now when I move the texture around with WASD it also corespondidly moves the image vectors around 

Scale

This is how it looks for scale fully zooomed in and zoomed out

<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/38acc3fe-56f6-49eb-86f5-96ce315d5a4e" />



<img width="1115" height="628" alt="Screenshot 2026-03-03 134547" src="https://github.com/user-attachments/assets/ac83f2f9-e7e1-4840-986f-bc7216b6fbb0" />

Code
```

```


Rotate

<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/4de20d6b-f92f-4a36-8496-925be7fc2199" />
Code
```

```
Position by orgin
Scale by orgin
Rotate by orgin 
<img width="1115" height="628" alt="image" src="https://github.com/user-attachments/assets/a550e73e-d9e6-4d22-9c2e-689f98003887" />


This is how it looks like with all of those three things done
Code
```

```

This is how it looks scaling something by location and then translate it by vectors

<img width="1426" height="631" alt="image" src="https://github.com/user-attachments/assets/c3817881-4fb3-4793-940d-0ee26f24ff61" />

Code
```

```



### REFLECTION





Exercise 4: Image smoothing (optional)


### REFLECTION






## Reflection



## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 3](CUDA-Lab3.md)
- [Cuda lab 5](CUDA-Lab5.md)

