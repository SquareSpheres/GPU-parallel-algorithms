#include "device_launch_parameters.h"
#include "ShiloachVishkin.cuh"
#include "CudaUtils.cuh"

namespace SV {
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

}

namespace SVU {

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

				if (fromVertex < toVertex)
				{
					*hasGrafted = true;
					component[toVertex] = fromVertex;
				}
			}

			{
				const int fromVertex = graph[id].second;
				const int toVertex = graph[id].first;

				if (fromVertex < toVertex && toVertex == component[toVertex])
				{
					*hasGrafted = true;
					component[toVertex] = fromVertex;
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


	__global__ void updateKernel(std::pair<int, int> *graph, const int numEdges, int *component)
	{
		unsigned int id = getGlobalIdx_3D_3D();
		unsigned int numThreads = getNumThreads();

		for (; id < numEdges; id += numThreads)
		{
			graph[id].first = component[graph[id].first];
			graph[id].second = component[graph[id].second];
		}
	}
}


int* ShiloachVishkinUpdt(std::pair<int, int> *graph, const int numVertices, const int numEdges)
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
	//TODO USE PINNED MEMORY

	// transfer data from host to device
	cudaMemcpy(d_graph, graph, numBytesGraph, cudaMemcpyHostToDevice);


	// max number of blocks per dim = 65535
	// max number of threads per block = 1024
	// figure out optimal block size, and gridsize

	dim3 threadsPerBlock(1024, 1); // 1024
	dim3 numBlocks((numEdges + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);


	//std::cout << "numThreadsPerBlock = " << threadsPerBlock.x << endl;
	//std::cout << "numBlock = " << numBlocks.x << endl;

	SVU::initComponentKernel << <numBlocks, threadsPerBlock >> > (d_results, numVertices);

	while (*h_hasGrafted)
	{
		*h_hasGrafted = false;
		// execute graft kernel
		SVU::graftKernel << <numBlocks, threadsPerBlock >> > (d_graph, numEdges, d_results, d_hasGrafted);
		// execute shortcut kernel
		SVU::shortcutKernel << <numBlocks, threadsPerBlock >> > (d_results, numVertices);
		// execute update kernel
		SVU::updateKernel << <numBlocks, threadsPerBlock >> > (d_graph, numEdges, d_results);
		// check if has grafted
		cudaMemcpy(h_hasGrafted, d_hasGrafted, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	// copy kernel result back to host side
	cudaMemcpy(h_results, d_results, numBytesResult, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_graph);
	cudaFree(d_results);
	cudaFree(d_hasGrafted);

	// free host memory
	delete h_hasGrafted;

	return h_results;
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
	//TODO USE PINNED MEMORY

	// transfer data from host to device
	cudaMemcpy(d_graph, graph, numBytesGraph, cudaMemcpyHostToDevice);


	// max number of blocks per dim = 65535
	// max number of threads per block = 1024
	// figure out optimal block size, and gridsize

	dim3 threadsPerBlock(1024, 1); // 1024
	dim3 numBlocks((numEdges + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);


	//std::cout << "numThreadsPerBlock = " << threadsPerBlock.x << endl;
	//std::cout << "numBlock = " << numBlocks.x << endl;

	SV::initComponentKernel << <numBlocks, threadsPerBlock >> > (d_results, numVertices);

	while (*h_hasGrafted)
	{
		*h_hasGrafted = false;
		// execute graft kernel
		SV::graftKernel << <numBlocks, threadsPerBlock >> > (d_graph, numEdges, d_results, d_hasGrafted);
		// execute shortcut kernel
		SV::shortcutKernel << <numBlocks, threadsPerBlock >> > (d_results, numVertices);
		// check if has grafted
		cudaMemcpy(h_hasGrafted, d_hasGrafted, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	// copy kernel result back to host side
	cudaMemcpy(h_results, d_results, numBytesResult, cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_graph);
	cudaFree(d_results);
	cudaFree(d_hasGrafted);

	// free host memory
	delete h_hasGrafted;

	return h_results;
}