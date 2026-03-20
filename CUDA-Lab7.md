# CUDA Lab 7 - A simple CUDA ray caster

[Cuda lab 6](CUDA-Lab6.md) | [Next: Cuda lab 8  →](CUDA-Lab8.md)


## SMART Objectives
1. Understand how ray casting works  
2. Learn how to implement a simple ray-caster 


## Completed Tasks

Exercise 1. Drawing based on a canvas of size [-1, 1]x[-1, 1] 


<img width="498" height="528" alt="image" src="https://github.com/user-attachments/assets/d9f5b5e2-bd09-422d-b3d2-2a289812dc90" />


Code
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
            d_output[i] = make_uchar4(0, 0, 0xff, 0xff); // red circle
        }
        else {
            d_output[i] = make_uchar4(0, 0, 0, 0xff);   // black background
        }
    }
}
```

## Reflection

This is similar to what we have done before for the circle but instead in a specific canvas 

   
Exercise 2. Write a simple ray caster 

Image here 
```
<img width="498" height="528" alt="image" src="https://github.com/user-attachments/assets/63a12471-a8fb-4ba1-9ff1-bfbeb03f0f62" />

```

Code from the cu for rendering stuff

```CUDA

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cfloat>

#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_cuda.h>

#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include "sphere.h"
#include "hitable_list.h"

typedef unsigned int uint;
typedef unsigned char uchar;

__device__ vec3 castRay(const ray& r, hitable** world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0f, FLT_MAX, rec)) {
		return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
	}
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void create_world(hitable** d_list, hitable** d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
		d_list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
		*d_world = new hitable_list(d_list, 2);
	}
}

__global__ void d_render(uchar4* d_output, uint width, uint height, hitable** d_world) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;

	if (x >= width || y >= height) {
		return;
	}

	float u = x / (float)width;
	float v = y / (float)height;

	u = 2.0f * u - 1.0f;
	v = -(2.0f * v - 1.0f);
	u *= width / (float)height;

	u *= 2.0f;
	v *= 2.0f;

	vec3 eye(0.0f, 0.5f, 1.5f);
	float distFrEye2Img = 1.0f;

	vec3 pixelPos(u, v, eye.z() - distFrEye2Img);
	ray r(eye, pixelPos - eye);

	vec3 col = castRay(r, d_world);
	d_output[i] = make_uchar4(col.x() * 255.0f, col.y() * 255.0f, col.z() * 255.0f, 0);
}

extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4* output) {
	hitable** d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hitable*)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));

	create_world<<<1, 1>>>(d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	d_render<<<gridSize, blockSize>>>(output, width, height, d_world);
	getLastCudaError("kernel failed");
}

#endif

```

Code from the cpp

```
/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * Neither the name of NVIDIA CORPORATION nor the names of its
* contributors may be used to endorse or promote products derived
* from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>
// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// CUDA system and GL includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// Helper functions
#include <helper_functions.h> // CUDA SDK Helper functions
#include <helper_cuda.h> // CUDA device initialization helper functions

typedef unsigned int uint;
typedef unsigned char uchar;

#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif

StopWatchInterface* timer = 0;
bool g_Verify = false;
int* pArgc = NULL;
char** pArgv = NULL;
#define REFRESH_DELAY 10 // ms
uint width = 512, height = 512;
uint imageWidth, imageHeight;
dim3 blockSize(32, 32);
dim3 gridSize(width / blockSize.x, height / blockSize.y);
GLuint pbo = 0; // OpenGL pixel buffer object
struct cudaGraphicsResource* cuda_pbo_resource; // handles OpenGL-CUDA exchange
GLuint displayTex = 0;
GLuint bufferTex = 0;
void display();
void initGLBuffers();
void cleanup();
#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB
extern "C" void initGL(int* argc, char** argv);

extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4* output);




// display results using OpenGL (called by GLUT)
void display() {
	sdkStartTimer(&timer);
	// map PBO to get CUDA device pointer
	uchar4* d_output;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
		(void**)&d_output, &num_bytes, cuda_pbo_resource));
	render(imageWidth, imageHeight, blockSize, gridSize,
		d_output);
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
	// Common display path
	{
		// display results
		glClear(GL_COLOR_BUFFER_BIT);
		// download image from PBO to OpenGL texture
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture(GL_TEXTURE_TYPE, displayTex);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexSubImage2D(GL_TEXTURE_TYPE, 0, 0, 0, width, height, GL_BGRA,
			GL_UNSIGNED_BYTE, 0);
		glEnable(GL_TEXTURE_TYPE);
		// draw textured quad
		glDisable(GL_DEPTH_TEST);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, (GLfloat)height);
		glVertex2f(0.0f, 0.0f);
		glTexCoord2f((GLfloat)width, (GLfloat)height);
		glVertex2f(1.0f, 0.0f);
		glTexCoord2f((GLfloat)width, 0.0f);
		glVertex2f(1.0f, 1.0f);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(0.0f, 1.0f);
		glEnd();
		glDisable(GL_TEXTURE_TYPE);
		glDisable(GL_FRAGMENT_PROGRAM_ARB);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	}
	glutSwapBuffers();
	glutReportErrors();
	sdkStopTimer(&timer);
}
// GLUT callback functions
void timerEvent(int value) {
	if (glutGetWindow()) {
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
	switch (key) {
	case 27:
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	default:
		break;
	}
}
void reshape(int x, int y) {
	width = x;
	height = y;
	imageWidth = width;
	imageHeight = height;
	initGLBuffers();
	glViewport(0, 0, x, y);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}
void cleanup() {
	//freeTexture();
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	glDeleteBuffers(1, &pbo);
	glDeleteTextures(1, &displayTex);
	sdkDeleteTimer(&timer);
}
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
void initGLBuffers() {
	if (pbo) {
		// delete old buffer
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
		glDeleteBuffers(1, &pbo);
	}
	// create pixel buffer object for display
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), 0,
		GL_STREAM_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(
		&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
	// create texture for display
	if (displayTex) {
		glDeleteTextures(1, &displayTex);
	}
	glGenTextures(1, &displayTex);
	glBindTexture(GL_TEXTURE_TYPE, displayTex);
	glTexImage2D(GL_TEXTURE_TYPE, 0, GL_RGBA8, width, height, 0, GL_RGBA,
		GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_TYPE, 0);
	// calculate new grid size
	gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}
void mainMenu(int i) { keyboard(i, 0, 0); }
void initMenus() {
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Quit [esc]", 27);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}
GLuint compileASMShader(GLenum program_type, const char* code) {
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
		(GLsizei)strlen(code), (GLubyte*)code);
	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
	if (error_pos != -1) {
		const GLubyte* error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos,
			error_string);
		return 0;
	}
	return program_id;
}
void initialize(int argc, char** argv) {
	initGL(&argc, argv);
	// use command-line specified CUDA device, otherwise use device with highest
	// Gflops/s
	int devID = findCudaDevice(argc, (const char**)argv);
	// get number of SMs on this GPU
	cudaDeviceProp deviceProps;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
	printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name,
		deviceProps.multiProcessorCount);
	// Create the timer (for fps measurement)
	sdkCreateTimer(&timer);
	printf(
		"\t[esc] - Quit\n\n"
	);
	initGLBuffers();
}
void initGL(int* argc, char** argv) {
	// initialize GLUT callback functions
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA bicubic texture filtering");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutCloseFunc(cleanup);
	initMenus();
	if (!isGLVersionSupported(2, 0) ||
		!areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "Required OpenGL extensions are missing.");
		exit(EXIT_FAILURE);
	}
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	pArgc = &argc;
	pArgv = argv;
	// parse arguments
	char* filename;
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif
	printf("Starting Original Texture\n");
	// This runs the CUDA kernel (bicubicFiltering) + OpenGL visualization
	initialize(argc, argv);
	glutMainLoop();
	exit(EXIT_SUCCESS);
}

//need to have the h files working since they dont fsr
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int> (result) << "at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting 
		cudaDeviceReset();
		exit(99);
	}
}

```

this is how it looks with 10 objects

output

for whatever reason it only loads 7 out of 10 objects it could be due to some of it being out of view of the camera?

```
<img width="3440" height="1392" alt="image" src="https://github.com/user-attachments/assets/ed3d72f6-1bd3-4742-84d1-bc2e472ff948" />

```


Code


```
/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cfloat>

#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_cuda.h>

#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include "sphere.h"
#include "hitable_list.h"

typedef unsigned int uint;
typedef unsigned char uchar;

__device__ vec3 castRay(const ray& r, hitable** world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0f, FLT_MAX, rec)) {
		return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
	}
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void create_world(hitable** d_list, hitable** d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f); //first bit is the x coords second the y coords final one is z maybe?
		d_list[1] = new sphere(vec3(-2.0f, -0.5f, -1.0f), 0.3f);
		d_list[2] = new sphere(vec3(-1.8f, -0.7f, -1.0f), 1.0f);
		d_list[3] = new sphere(vec3(1.4f, -0.6f, -1.0f), 0.2f);
		d_list[4] = new sphere(vec3(3.0f, -0.8f, -1.0f), 0.1f);
		d_list[5] = new sphere(vec3(5.0f, -0.1f, -1.0f), 0.3f);
		d_list[6] = new sphere(vec3(4.0f, -0.5f, -1.0f), 1.0f);
		d_list[7] = new sphere(vec3(3.5f, -0.4f, -1.0f), 0.6f);
		d_list[8] = new sphere(vec3(-1.0f, -0.3f, -1.0f), 0.8f);
		d_list[9] = new sphere(vec3(1.0f, -0.2f, -1.0f), 0.7f);

		*d_world = new hitable_list(d_list, 10);
	}
}

__global__ void d_render(uchar4* d_output, uint width, uint height, hitable** d_world) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;

	if (x >= width || y >= height) {
		return;
	}

	float u = x / (float)width;
	float v = y / (float)height;

	u = 2.0f * u - 1.0f;
	v = -(2.0f * v - 1.0f);
	u *= width / (float)height;

	u *= 2.0f;
	v *= 2.0f;

	vec3 eye(0.0f, 0.5f, 1.5f);
	float distFrEye2Img = 1.0f;

	vec3 pixelPos(u, v, eye.z() - distFrEye2Img);
	ray r(eye, pixelPos - eye);

	vec3 col = castRay(r, d_world);
	d_output[i] = make_uchar4(col.z() * 255.0f, col.y() * 255.0f, col.x() * 255.0f, 0);
}

extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4* output) {
	hitable** d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hitable*)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));

	create_world<<<1, 1>>>(d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	d_render<<<gridSize, blockSize>>>(output, width, height, d_world);
	getLastCudaError("kernel failed");
}

#endif
```

## Reflection



## Beyond the Lab (Optional)


**Navigation:**
- [Cuda lab 6](CUDA-Lab6.md)
- [Cuda lab 8](CUDA-Lab8.md)

