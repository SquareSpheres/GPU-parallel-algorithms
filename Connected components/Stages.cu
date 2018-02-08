#include "device_launch_parameters.h"
#include "Stages.cuh"
#include "CudaUtils.cuh"

namespace sv {

	__device__ bool has_grafted_d = true;

	__global__ void InitComponentKernel(int *component, const int numVertices)
	{
		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		for (auto i = tid; i < numVertices; i += numThreads) {
			component[i] = i;
		}
	}

	__global__ void GraftKernel(std::pair<int, int> *graph, const int numEdges, int *component)
	{

		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		has_grafted_d = false;

		for (auto i = tid; i < numEdges; i += numThreads)
		{

			int fromVertex = graph[i].first;
			int toVertex = graph[i].second;

			int fromComponent = component[fromVertex];
			int toComponent = component[toVertex];

			if ((fromComponent < toComponent) && (toComponent == component[toComponent]))
			{
				has_grafted_d = true;
				component[toComponent] = fromComponent;

			}


			const int tmp = fromVertex;
			fromVertex = toVertex;
			toVertex = tmp;

			fromComponent = component[fromVertex];
			toComponent = component[toVertex];

			if ((fromComponent < toComponent) && (toComponent == component[toComponent]))
			{
				has_grafted_d = true;
				component[toComponent] = fromComponent;
			}
		}
	}

	__global__ void ShortcutKernel(int *component, const int numVertices)
	{

		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		for (auto i = tid; i < numVertices; i += numThreads)
		{
			while (component[i] != component[component[i]])
			{
				component[i] = component[component[i]];
			}
		}
	}

}


namespace svu {

	__device__ bool has_grafted_d = false;

	__global__ void InitComponentKernel(int *component, const int numVertices)
	{
		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		for (auto i = tid; i < numVertices; i += numThreads) {
			component[i] = i;
		}
	}

	__global__ void GraftKernel(std::pair<int, int> *graph, const int numEdges, int *component)
	{

		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		has_grafted_d = false;

		for (auto i = tid; i < numEdges; i += numThreads)
		{

			int fromVertex = graph[i].first;
			int toVertex = graph[i].second;

			if (fromVertex < toVertex)
			{
				has_grafted_d = true;
				component[toVertex] = fromVertex;
			}

			const int tmp = fromVertex;
			fromVertex = toVertex;
			toVertex = tmp;

			if (fromVertex < toVertex)
			{
				has_grafted_d = true;
				component[toVertex] = fromVertex;
			}
		}
	}

	__global__ void ShortcutKernel(int *component, const int numVertices)
	{

		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		for (auto i = tid; i < numVertices; i += numThreads)
		{
			while (component[i] != component[component[i]])
			{
				component[i] = component[component[i]];
			}
		}
	}

	__global__ void UpdateKernel(std::pair<int, int> *graph, const int numEdges, int *component)
	{
		const auto tid = blockIdx.x*blockDim.x + threadIdx.x;
		const auto numThreads = gridDim.x * blockDim.x;

		for (auto i = tid; i < numEdges; i += numThreads)
		{
			graph[i].first = component[graph[i].first];
			graph[i].second = component[graph[i].second];
		}
	}
}


void ShiloachVishkinUpdt(std::pair<int, int> *graph, const int numEdges, int *components, const int numVertices)
{

	//Set caching prefrence, prefer L1
	CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// init device memory
	std::pair<int, int> *d_graph = nullptr;
	int *d_results = nullptr;

	const int numBytesGraph = numEdges * sizeof(std::pair<int, int>);
	const int numBytesResult = numVertices * sizeof(int);


	CHECK(cudaMalloc((void **)&d_graph, numBytesGraph));
	CHECK(cudaMalloc((void **)&d_results, numBytesResult));

	// has grafted
	bool has_grafted_h = true;


	// transfer data from host to device
	CHECK(cudaMemcpy(d_graph, graph, numBytesGraph, cudaMemcpyHostToDevice));


	int threads_per_block = 1024;
	int blocks_per_grid = 30;

	svu::InitComponentKernel << <blocks_per_grid, threads_per_block >> > (d_results, numVertices);
	//synchronize with host
	CHECK(cudaDeviceSynchronize());
	// check for errors
	CHECK(cudaGetLastError());

	while (has_grafted_h)
	{
		// execute graft kernel
		svu::GraftKernel << <blocks_per_grid, threads_per_block >> > (d_graph, numEdges, d_results);
		//synchronize with host
		CHECK(cudaDeviceSynchronize());
		// check for errors
		CHECK(cudaGetLastError());
		// execute shortcut kernel
		svu::ShortcutKernel << <blocks_per_grid, threads_per_block >> > (d_results, numVertices);
		//synchronize with host
		CHECK(cudaDeviceSynchronize());
		// check for errors
		CHECK(cudaGetLastError());
		// execute update kernel
		svu::UpdateKernel << <blocks_per_grid, threads_per_block >> > (d_graph, numEdges, d_results);
		//synchronize with host
		CHECK(cudaDeviceSynchronize());
		// check for errors
		CHECK(cudaGetLastError());
		// check if has grafted
		CHECK(cudaMemcpyFromSymbol(&has_grafted_h, svu::has_grafted_d, sizeof(bool), 0, cudaMemcpyDeviceToHost));
	}

	// copy kernel result back to host side
	CHECK(cudaMemcpy(components, d_results, numBytesResult, cudaMemcpyDeviceToHost));


	// free device memory
	CHECK(cudaFree(d_graph));
	CHECK(cudaFree(d_results));
}


void ShiloachVishkin(std::pair<int, int> *graph, const int numEdges, int *components, const int numVertices)
{

	//Set caching prefrence, prefer L1
	CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// init device memory
	std::pair<int, int> *d_graph = nullptr;
	int *d_results = nullptr;

	const int numBytesGraph = numEdges * sizeof(std::pair<int, int>);
	const int numBytesResult = numVertices * sizeof(int);

	CHECK(cudaMalloc((void **)&d_graph, numBytesGraph));
	CHECK(cudaMalloc((void **)&d_results, numBytesResult));

	// has grafted
	bool has_grafted_h = true;

	// transfer data from host to device
	CHECK(cudaMemcpy(d_graph, graph, numBytesGraph, cudaMemcpyHostToDevice));

	// max number of blocks per dim = 65535
	// max number of threads per block = 1024
	// figure out optimal block size, and gridsize
	// (numEdges + threadsPerBlock.x - 1) / threadsPerBlock.x

	// (15) Multiprocessors, (128) CUDA Cores/MP:     1920 CUDA Cores
	// Maximum number of threads per multiprocessor:  2048
	// Maximum number of threads per block:           1024
	// Max dimension size of a thread block(x, y, z): (1024, 1024, 64)
	// Max dimension size of a grid size(x, y, z):    (2147483647, 65535, 65535)

	int threads_per_block = 1024;
	int blocks_per_grid = 30;

	sv::InitComponentKernel << <blocks_per_grid, threads_per_block >> > (d_results, numVertices);
	//synchronize with host
	CHECK(cudaDeviceSynchronize());
	// check for errors
	CHECK(cudaGetLastError());

	while (has_grafted_h)
	{
		// execute graft kernel
		sv::GraftKernel << <blocks_per_grid, threads_per_block >> > (d_graph, numEdges, d_results);
		//synchronize with host
		CHECK(cudaDeviceSynchronize());
		// check for errors
		CHECK(cudaGetLastError());
		// execute shortcut kernel
		sv::ShortcutKernel << <blocks_per_grid, threads_per_block >> > (d_results, numVertices);
		//synchronize with host
		CHECK(cudaDeviceSynchronize());
		// check for errors
		CHECK(cudaGetLastError());
		// check if hasGrafted
		CHECK(cudaMemcpyFromSymbol(&has_grafted_h, sv::has_grafted_d, sizeof(bool), 0, cudaMemcpyDeviceToHost));
	}

	// copy kernel result back to host side
	CHECK(cudaMemcpy(components, d_results, numBytesResult, cudaMemcpyDeviceToHost));

	// free device memory
	CHECK(cudaFree(d_graph));
	CHECK(cudaFree(d_results));
}

void Stages(std::pair<int, int>* graph, const int numEdges, int * components, const int numVertices)
{


	//randomly permute El
	//std::random_device randomDevice;
	//std::mt19937 engine(randomDevice());
	//std::shuffle(graph, graph + numEdges, engine);


	// divide El into groups
	int numberOfGroups = 1;
	int numberofEdgesPerGroup = numEdges;
	const double q = 0.6;
	const double numberOfEdgesPerVertex = numEdges / static_cast<double>(numVertices);

	
	if (q < numberOfEdgesPerVertex)
	{
		numberofEdgesPerGroup = q*numVertices;
		numberOfGroups = ceil(numEdges / static_cast<double>(numberofEdgesPerGroup));
	}

	//Set caching prefrence, prefer L1
	CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// init device memory
	std::pair<int, int> *d_graph = nullptr;
	int *d_results = nullptr;

	const int numBytesGraph = numEdges * sizeof(std::pair<int, int>);
	const int numBytesResult = numVertices * sizeof(int);


	CHECK(cudaMalloc((void **)&d_graph, numBytesGraph));
	CHECK(cudaMalloc((void **)&d_results, numBytesResult));

	// transfer data from host to device
	CHECK(cudaMemcpy(d_graph, graph, numBytesGraph, cudaMemcpyHostToDevice));


	int threads_per_block = 1024;
	int blocks_per_grid = 30;

	// initialize components
	svu::InitComponentKernel << <blocks_per_grid, threads_per_block >> > (d_results, numVertices);
	// synchronize with host
	CHECK(cudaDeviceSynchronize());
	// check for errors
	CHECK(cudaGetLastError());

	

	for (int i = 0; i < numberOfGroups; ++i)
	{
		const int groupStart = numberofEdgesPerGroup*i;
		int currentNumberofEdgesPerGroup = numberofEdgesPerGroup;
		if (i == numberOfGroups - 1) { currentNumberofEdgesPerGroup = numEdges - (numberOfGroups - 1)*numberofEdgesPerGroup; }

		// has grafted
		bool has_grafted_h = true;

		while (has_grafted_h)
		{
			// execute graft kernel
			svu::GraftKernel << <blocks_per_grid, threads_per_block >> > (d_graph + groupStart, currentNumberofEdgesPerGroup, d_results);
			//synchronize with host
			CHECK(cudaDeviceSynchronize());
			// check for errors
			CHECK(cudaGetLastError());
			// execute shortcut kernel
			svu::ShortcutKernel << <blocks_per_grid, threads_per_block >> > (d_results, numVertices);
			//synchronize with host
			CHECK(cudaDeviceSynchronize());
			// check for errors
			CHECK(cudaGetLastError());
			// execute update kernel
			svu::UpdateKernel << <blocks_per_grid, threads_per_block >> > (d_graph + groupStart, currentNumberofEdgesPerGroup, d_results);
			//synchronize with host
			CHECK(cudaDeviceSynchronize());
			// check for errors
			CHECK(cudaGetLastError());
			// check if has grafted
			CHECK(cudaMemcpyFromSymbol(&has_grafted_h, svu::has_grafted_d, sizeof(bool), 0, cudaMemcpyDeviceToHost));
		}


		if (i < (numberOfGroups - 1))
		{
			int numberOfEdgesNextGroup = currentNumberofEdgesPerGroup;
			// if second last group
			if(i==numberOfGroups-2) { numberOfEdgesNextGroup = numEdges - (numberOfGroups - 1)*numberofEdgesPerGroup; }
			// execute update kernel
			svu::UpdateKernel << <blocks_per_grid, threads_per_block >> > (d_graph + groupStart + currentNumberofEdgesPerGroup, numberOfEdgesNextGroup, d_results);
			//synchronize with host
			CHECK(cudaDeviceSynchronize());
			// check for errors
			CHECK(cudaGetLastError());
		}

	}

	// copy kernel result back to host side
	CHECK(cudaMemcpy(components, d_results, numBytesResult, cudaMemcpyDeviceToHost));

	// free device memory
	CHECK(cudaFree(d_graph));
	CHECK(cudaFree(d_results));

}