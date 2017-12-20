#pragma once
#include "cuda_runtime.h"
#include <utility>
#include <vector>


/**
* Guojing Cong and Paul Muzio Connected component algorithms
*
* Paper : https://link.springer.com/chapter/10.1007/978-3-319-14325-5_14
*
*/

namespace sv {
	/**
	* Shortcut vertives in same components to a single super-vertex
	*/
	__global__ void ShortcutKernel(int *component, const int numVertices);
	/**
	* Each processor inspects an edge, and tries to graft the larger endpoint (by index) to the smaller one
	*/
	__global__ void GraftKernel(std::pair<int, int> *graph, const int numEdges, int *component);
	/**
	* Initialize the component array. Initially each vertex is its own component
	*/
	__global__ void InitComponentKernel(int *component, const int numVertices);
}

namespace svu {
	/**
	* Shortcut vertives in same components to a single super-vertex
	*/
	__global__ void ShortcutKernel(int *component, const int numVertices);
	/**
	* Each processor inspects an edge, and tries to graft the larger endpoint (by index) to the smaller one
	*/
	__global__ void GraftKernel(std::pair<int, int> *graph, const int numEdges, int *component);
	/**
	* Initialize the component array. Initially each vertex is its own component
	*/
	__global__ void InitComponentKernel(int *component, const int numVertices);
	/**
	* Replaces each edge (u, v) with (C[u], C[v])
	*/
	__global__ void UpdateKernel(std::pair<int, int> *graph, const int numEdges, int *component);
}

std::vector<int> ShiloachVishkin(std::pair<int, int> *graph, const int numVertices, const int numEdges);
std::vector<int> ShiloachVishkinUpdt(std::pair<int, int> *graph, const int numVertices, const int numEdges);