
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define W 500
#define H 500
#define TX 32
#define TY 32

__device__
float distance(float2 ref, float c, float r)
{
	float dist = sqrtf((c - ref.x) * (c - ref.x) + (r - ref.y) * (r - ref.y));
	return dist;
}

__global__
void distanceKernel(float* d_out, int w, int h, float2 pos)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = r * w + c;

	if ((c >= w) || (r >= h))
	{
		return;
	}

	float dist = distance(pos, c, r);
	d_out[i] = dist;

	printf("i = %2d: Start pos = (%f, %f), End pos = (%i, %i), Distance = %f\n", i, pos.x, pos.y, c, r, dist);
}

int main()
{
	float* out = (float*) calloc(W * H, sizeof(float));
	float* d_out;
	cudaMalloc(&d_out, W * H * sizeof(float));

	const float2 pos = { 0.0f, 0.0f };
	const dim3 blockSize(TX, TY);
	const int bx = (W + TX - 1) / TX;
	const int by = (W + TY - 1) / TY;
	const dim3 gridSize = dim3(bx, by);

	distanceKernel << <gridSize, blockSize >> > (d_out, W, H, pos);

	cudaMemcpy(out, d_out, W * H * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	free(out);
	return 0;
}
