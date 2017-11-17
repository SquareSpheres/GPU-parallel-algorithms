#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }


// Courtesy of "talonmies" https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ unsigned int GetGlobalIdx_3D_3D()
{
	const int blockId = blockIdx.x
		+ blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	const int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x)
		+ threadIdx.x;
	return threadId;

}

__device__ unsigned int GetNumThreads()
{
	const unsigned int numBlocks = gridDim.x * gridDim.y * gridDim.z;
	const unsigned int numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	return numBlocks * numThreadsPerBlock;
}
