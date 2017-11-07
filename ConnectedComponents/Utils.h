#pragma once

#include <vector>
#include <utility>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

using namespace std;
using namespace boost;


vector<int> BoostConnectedComponent(adjacency_list<vecS, vecS, undirectedS> &graph);
void GenerateRandomGraphToFile(const std::string filename, const int numVertices, const float prob);
adjacency_list<vecS, vecS, undirectedS> GenerateRandomGraphBoost(const int numVertices, const float prob);
std::pair<int, int> ReadGraphFromFile(const std::string filename, std::vector<std::pair<int, int>>* buffer);
std::pair<int, int> FromBoostToStdGraph(adjacency_list<vecS, vecS, undirectedS> &boostGraph, std::vector<std::pair<int, int>>* buffer);
std::pair<int, int> FromBoostToStdGraphBi(adjacency_list<vecS, vecS, undirectedS> &boostGraph, std::vector<std::pair<int, int>>* buffer);
void PrintArray_int(int *arr, const unsigned int size);
void PrintVector_int(std::vector<int> &vec);
void PostProcessConnectedCompnent(int *component, const unsigned int numVertices);


