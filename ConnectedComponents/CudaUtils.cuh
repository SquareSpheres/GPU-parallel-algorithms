#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ unsigned int getGlobalIdx_3D_3D()
{
	int blockId = blockIdx.x
		+ blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x)
		+ threadIdx.x;
	return threadId;
}

__device__ unsigned int getNumThreads()
{
	unsigned int numBlocks = gridDim.x * gridDim.y * gridDim.z;
	unsigned int numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
	return numBlocks * numThreadsPerBlock;
}
