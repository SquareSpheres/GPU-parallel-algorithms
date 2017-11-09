#pragma once
#include "cuda_runtime.h"
#include <utility>

namespace SV {
	__global__ void shortcutKernel(int *component, int numVertices);
	__global__ void graftKernel(std::pair<int, int> *graph, const int numEdges, int *component, bool *hasGrafted);
	__global__ void initComponentKernel(int *component, const int numVertices);
}

namespace SVU {
	__global__ void shortcutKernel(int *component, int numVertices);
	__global__ void graftKernel(std::pair<int, int> *graph, const int numEdges, int *component, bool *hasGrafted);
	__global__ void initComponentKernel(int *component, const int numVertices);
	__global__ void updateKernel(std::pair<int, int> *graph, const int numEdges, int *component);
}

int* ShiloachVishkin(std::pair<int, int> *graph, const int numVertices, const int numEdges);
int* ShiloachVishkinUpdt(std::pair<int, int> *graph, const int numVertices, const int numEdges);