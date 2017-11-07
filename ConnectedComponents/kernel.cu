#include "Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <utility>
#include <stdio.h>
#include <omp.h>

__device__ unsigned int getNumThreads();
__device__ unsigned int getGlobalIdx_3D_3D();
__global__ void shortcutKernel(int *component, int numVertices);
__global__ void graftKernel(std::pair<int, int> *graph, const int numEdges, int *component, bool *hasGrafted);
__global__ void initComponentKernel(int *component, const int numVertices);




__global__ void initComponentKernel(int *component, const int numVertices)
{
	unsigned int id = getGlobalIdx_3D_3D();
	unsigned int numThreads = getNumThreads();

	for (; id < numVertices; id += numThreads) {
		component[id] = id;
	}
}


__global__ void graftKernel(std::pair<int, int> *graph, const int numEdges, int *component, bool *hasGrafted)
{

	unsigned int id = getGlobalIdx_3D_3D();
	unsigned int numThreads = getNumThreads();

	*hasGrafted = false;

	for (; id < numEdges; id += numThreads)
	{
		{
			const int fromVertex = graph[id].first;
			const int toVertex = graph[id].second;

			const int fromComponent = component[fromVertex];
			const int toComponent = component[toVertex];

			if ((fromComponent < toComponent))
			{
				*hasGrafted = true;
				component[toComponent] = fromComponent;
			}
		}

		{
			const int fromVertex = graph[id].second;
			const int toVertex = graph[id].first;

			const int fromComponent = component[fromVertex];
			const int toComponent = component[toVertex];

			if ((fromComponent < toComponent))
			{
				*hasGrafted = true;
				component[toComponent] = fromComponent;
			}
		}
	}

}

__global__ void shortcutKernel(int *component, int numVertices)
{

	unsigned int id = getGlobalIdx_3D_3D();
	unsigned int numThreads = getNumThreads();

	for (; id < numVertices; id += numThreads)
	{
		while (component[id] != component[component[id]])
		{
			component[id] = component[component[id]];
		}
	}
}


int* ShiloachVishkin(std::pair<int, int> *graph, const int numVertices, const int numEdges)
{

	// init device memory
	std::pair<int, int> *d_graph = 0;
	int *d_results = 0;
	bool *d_hasGrafted = 0;

	int numBytesGraph = numEdges * sizeof(std::pair<int, int>);
	int numBytesResult = numVertices * sizeof(int);

	cudaMalloc((void **)&d_graph, numBytesGraph);
	cudaMalloc((void **)&d_results, numBytesResult);
	cudaMalloc((void **)&d_hasGrafted, sizeof(bool));

	// init host memory for results
	int *h_results = new int[numVertices];
	bool *h_hasGrafted = new bool{ true };

	// transfer data from host to device
	cudaMemcpy(d_graph, graph, numBytesGraph, cudaMemcpyHostToDevice);


	// max number of blocks per dim = 65535
	// max number of threads per block = 1024
	// figure out optimal block size, and gridsize
	dim3 threadsPerBlock(256, 1); // 128
	dim3 numBlocks((numEdges + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);


	std::cout << "numThreadsPerBlock = " << threadsPerBlock.x << endl;
	std::cout << "numBlock = " << numBlocks.x << endl;

	initComponentKernel << <numBlocks, threadsPerBlock >> > (d_results, numVertices);

	while (*h_hasGrafted)
	{
		*h_hasGrafted = false;
		// execute graft kernel
		graftKernel << <numBlocks, threadsPerBlock >> > (d_graph, numEdges, d_results, d_hasGrafted);
		// execute shortcut kernel
		shortcutKernel << <numBlocks, threadsPerBlock >> > (d_results, numVertices);
		// check if has grafted
		cudaMemcpy(h_hasGrafted, d_hasGrafted, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	// copy kernel result back to host side
	cudaMemcpy(h_results, d_results, numBytesResult, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_graph);
	cudaFree(d_results);
	// free host memory
	delete h_hasGrafted;

	return h_results;
}


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



int main()
{

	typedef vector<std::pair<int, int>> StdGraph;
	typedef adjacency_list<vecS, vecS, undirectedS> BoostGraph;

	BoostGraph boostGraph = GenerateRandomGraphBoost(40, 0.12f);
	StdGraph stdGraph;

	const std::pair<int, int> numVertEdg = FromBoostToStdGraph(boostGraph, &stdGraph);

	std::cout << "numVertices = " << numVertEdg.first << endl;
	std::cout << "numEdges = " << numVertEdg.second << endl;

	const int numVerticesStd = numVertEdg.first;
	const int numEdgesStd = numVertEdg.second;

	const double time = omp_get_wtime();
	int *grapg = ShiloachVishkin(&stdGraph[0], numVerticesStd, numEdgesStd);
	const double timeEnd = omp_get_wtime() - time;

	const double boostTime = omp_get_wtime();
	std::vector<int> results = BoostConnectedComponent(boostGraph);
	const double boostTimeEnd = omp_get_wtime() - boostTime;

	// post process components
	PostProcessConnectedCompnent(grapg, numVerticesStd);

	if (!std::equal(grapg, grapg + numVerticesStd, results.begin(), results.end())) {
		std::cout << "Results does not match!" << endl;
		if (numVerticesStd < 101) {
			PrintArray_int(grapg, numVerticesStd);
			PrintVector_int(results);
		}
	}

	cudaDeviceReset();
	std::cout << "boost time = " << boostTimeEnd << endl;
	std::cout << "GPU time   = " << timeEnd << endl;

	return 0;
}
