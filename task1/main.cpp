#include <iostream>
#include <iomanip>
#include <cassert>

#include "../common/structures.hpp"
#include "../common/gnn.hpp"

using R = double;

using vector = PFN_intern_2019::vector<R>;
using matrix = PFN_intern_2019::matrix<R>;
using graph = PFN_intern_2019::graph<R>;
using PFN_intern_2019::calcFeatureVector;

const int N = 4;
const int T = 2;
const int D = 3;

int main(){
	// test graph
	graph G(N);
	G.addEdge(0,1);
	G.addEdge(0,2);
	G.addEdge(1,2);
	G.addEdge(0,3);
	// test matrix
	matrix W(D,D);
	W(0,0) = 0.3;	W(0,1) =-0.1;	W(0,2) = 0.4;
	W(1,0) =-0.1;	W(1,1) = 0.5;	W(1,2) = 0.9;
	W(2,0) = 0.2;	W(2,1) =-0.6;	W(2,2) = 0.5;
	// calculate
	vector feature = calcFeatureVector(G,T,W);
	// output
	std::cout << "feature vector" << std::endl;
	for(int i=0; i<D; i++){
		std::cout << i << ": ";
		std::cout << std::fixed << std::setprecision(5) << feature(i);
		std::cout << std::endl;
	}
	// test
	assert(std::abs(feature(0) - 3.06) < 1e-9);
	assert(std::abs(feature(1) - 2.70) < 1e-9);
	assert(std::abs(feature(2) - 2.88) < 1e-9);

	return 0;
}
