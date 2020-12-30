#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define TX 32
#define TY 32
#define LEN 5.0f
#define TIME_STEP 0.005f
#define FINAL_TIME 10.0f

// scale coordinates onto [-LEN, LEN]
__device__
float scale(int i, int w)
{
	return 2 * LEN * (((1.0f * i) / w) - 0.5f);
}

// function for right hand side of y equation
__device__
float f(float x, float y, float param, float sys)
{
	if (sys == 1) return x - 2 * param * y; // negative stifness
	if (sys == 2) return -x + param * (1 - x * x) * y; // van der Pol
	return -x - 2 * param * y;
}

// Explicit euler solver
__device__ 
float2 euler(float x, float y, float dt, float tFinal, float param, float sys)
{
	float dx = 0.0f;
	float dy = 0.0f;
	for (float t = 0; t < tFinal; t += dt)
	{
		dx = dt * y;
		dy = dt * f(x, y, param, sys);
		x += dx;
		y += dy;
	}
	return make_float2(x, y);
}

// Clamps n to [0, 255]
__device__
unsigned char clip(int n)
{
	if (n > 255)
		n = 255;
	else if (n < 0)
		n = 0;
	return n;
}

__global__
void stabImageKernel(uchar4* d_out, int w, int h, float p, int s)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	if ((c >= w) || r >= h)
	{
		return;
	}
	const int i = c + r * w;

	const float x0 = scale(c, w);
	const float y0 = scale(r, h);
	const float dist_0 = sqrt(x0 * x0 + y0 * y0);
	const float2 pos = euler(x0, y0, TIME_STEP, FINAL_TIME, p, s);
	const float dist_f = sqrt(pos.x * pos.x + pos.y + pos.y);
	
	//assign colors based on distance from origin
	const float dist_r = dist_f / dist_0;
	d_out[i].x = clip(dist_r * 255);
	d_out[i].y = ((c == w / 2) || (r == h / 2)) ? 255 : 0;
	d_out[i].z = clip((1 / dist_r) * 255);
	d_out[i].w = 255;
}

void kernelLauncher(uchar4* d_out, int w, int h, float p, int s)
{
	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);
	stabImageKernel << <gridSize, blockSize >> > (d_out, w, h, p, s);
}
