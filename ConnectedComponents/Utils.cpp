#include "Utils.h"
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/connected_components.hpp>
#include <fstream>
#include <ctime>


using namespace std;
using namespace boost;



void PostProcessConnectedCompnent(int *component, const unsigned int numVertices) {
	// post processing. Inlcude this in runtime?
	std::unordered_map<int, int> uniqueComp;
	int count = 0;

	for (int i = 0; i < numVertices; i++)
	{
		int value = component[i];
		if (uniqueComp.find(value) == uniqueComp.end())
		{
			uniqueComp.insert({ value, count++ });
		}

		component[i] = uniqueComp[component[i]];
	}
}


void PrintVector_int(std::vector<int> &vec)
{
	std::stringstream stream;
	for (int i = 0; i < vec.size(); ++i)
	{
		if (i == 0) stream << "[" << vec[i] << ",";
		else if (i == vec.size() - 1) stream << vec[i] << "]";
		else 	stream << vec[i] << ",";

	}

	std::cout << stream.str() << endl;
}

void PrintArray_int(int *arr, const unsigned int size)
{
	std::stringstream stream;
	for (int i = 0; i < size; ++i)
	{
		if (i == 0) stream << "[" << arr[i] << ",";
		else if (i == size - 1) stream << arr[i] << "]";
		else 	stream << arr[i] << ",";

	}

	std::cout << stream.str() << endl;
}


vector<int> BoostConnectedComponent(adjacency_list<vecS, vecS, undirectedS> &graph)
{
	std::vector<int> component(num_vertices(graph));
	int num = connected_components(graph, &component[0]);
	return component;
}


adjacency_list<vecS, vecS, undirectedS> GenerateRandomGraphBoost(const int numVertices, const float prob)
{
	typedef adjacency_list<vecS, vecS, undirectedS> Graph;
	typedef erdos_renyi_iterator<boost::minstd_rand, Graph> ERGen;

	// random generator
	boost::minstd_rand gen;
	gen.seed(time(nullptr));
	Graph g(ERGen(gen, numVertices, prob), ERGen(), numVertices);
	return g;
}



void GenerateRandomGraphToFile(const std::string filename, const int numVertices, const float prob)
{
	typedef adjacency_list<vecS, vecS, undirectedS> Graph;
	typedef erdos_renyi_iterator<boost::minstd_rand, Graph> ERGen;

	// random generator
	boost::minstd_rand gen;
	gen.seed(time(nullptr));
	Graph g(ERGen(gen, numVertices, prob), ERGen(), numVertices);

	graph_traits<Graph>::edge_iterator ei, ei_end;


	std::ofstream outFile;
	outFile.open(filename);
	// print num vertives
	outFile << numVertices << endl;
	// print num edges
	outFile << num_edges(g) << endl;

	for (tie(ei, ei_end) = edges(g); ei != ei_end; ++ei)
	{
		outFile << source(*ei, g) << " " << target(*ei, g) << endl;
	}

}

std::pair<int, int> ReadGraphFromFile(const std::string filename, std::vector<std::pair<int, int>>* buffer)
{
	string line;
	ifstream myfile(filename);
	if (myfile.is_open())
	{
		getline(myfile, line);
		int numVertices = stoi(line);
		getline(myfile, line);
		int numEdges = stoi(line);

		while (getline(myfile, line))
		{

			auto split = line.find(" ");
			int from = stoi(line.substr(0, split));
			int to = stoi(line.substr(split, line.size()));
			buffer->push_back(std::pair<int, int>{from, to});

		}
		myfile.close();
		return std::pair<int, int>{numVertices, numEdges};
	}

	return std::pair<int, int>{0, 0};
}


std::pair<int, int> FromBoostToStdGraph(adjacency_list<vecS, vecS, undirectedS> &boostGraph, std::vector<std::pair<int, int>>* buffer)
{
	typedef adjacency_list<vecS, vecS, undirectedS> Graph;

	buffer->clear();
	buffer->reserve(num_edges(boostGraph));


	std::pair<int, int> numVerticesAndEdges{ num_vertices(boostGraph), num_edges(boostGraph) };

	graph_traits<Graph>::edge_iterator ei, ei_end;
	for (tie(ei, ei_end) = edges(boostGraph); ei != ei_end; ++ei)
	{
		buffer->push_back(std::pair<int, int>{source(*ei, boostGraph), target(*ei, boostGraph)});
	}

	return numVerticesAndEdges;

}

std::pair<int, int> FromBoostToStdGraphBi(adjacency_list<vecS, vecS, undirectedS>& boostGraph, std::vector<std::pair<int, int>>* buffer)
{

	typedef adjacency_list<vecS, vecS, undirectedS> Graph;

	buffer->clear();
	buffer->reserve(2 * num_edges(boostGraph));

	std::pair<int, int> numVerticesAndEdges{ num_vertices(boostGraph),2 * num_edges(boostGraph) };

	graph_traits<Graph>::edge_iterator ei, ei_end;
	for (tie(ei, ei_end) = edges(boostGraph); ei != ei_end; ++ei)
	{
		buffer->push_back(std::pair<int, int>{source(*ei, boostGraph), target(*ei, boostGraph)});
		buffer->push_back(std::pair<int, int> {target(*ei, boostGraph), source(*ei, boostGraph)});
	}

	return numVerticesAndEdges;
}
