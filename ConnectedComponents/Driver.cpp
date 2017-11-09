#include "ShiloachVishkin.cuh"
#include "Utils.h"
#include <stdio.h>
#include <omp.h>

int main()
{

	typedef vector<std::pair<int, int>> StdGraph;
	typedef adjacency_list<vecS, vecS, undirectedS> BoostGraph;

	BoostGraph boostGraph = GenerateRandomGraphBoost(4000000, 0.000002f);
	StdGraph stdGraph;

	const std::pair<int, int> numVertEdg = FromBoostToStdGraph(boostGraph, &stdGraph);

	std::cout << "numVertices = " << numVertEdg.first << endl;
	std::cout << "numEdges = " << numVertEdg.second << endl;

	const int numVerticesStd = numVertEdg.first;
	const int numEdgesStd = numVertEdg.second;
	
	const double time = omp_get_wtime();
	int *grapg = ShiloachVishkinUpdt(&stdGraph[0], numVerticesStd, numEdgesStd);
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