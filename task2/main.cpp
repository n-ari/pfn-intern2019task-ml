#include <iostream>
#include <random>

#include "../common/structures.hpp"
#include "../common/gnn.hpp"

using R = double;

using graph = PFN_intern_2019::graph<R>;
using PFN_intern_2019::calcLoss;
using PFN_intern_2019::calcLabel;
using PFN_intern_2019::calcGradient;
using Param = PFN_intern_2019::GNNParam<R>;

const int N = 10;
const int T = 2;
const int D = 8;
const R alpha = 0.0001;
const R epsilon = 0.001;

int main(){
	// RNG
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());

	// random graph and label
	graph G(N);
	bool y;
	{
		std::uniform_int_distribution<int> dist(0,100);
		for(int i=0; i<N; i++){
			for(int j=0; j<i; j++){
				if(dist(engine)<30){		// 30%
					G.addEdge(i,j);
				}
			}
		}
		y = dist(engine)<50;				// 50%
	}

	// initial params
	Param theta(D);
	{
		std::normal_distribution<R> dist(0.0, 0.4);
		for(int i=0; i<D; i++){
			for(int j=0; j<D; j++){
				theta.W(i,j) = dist(engine);
			}
		}
		for(int i=0; i<D; i++){
			theta.A(i) = dist(engine);
		}
		theta.b = 0;
	}

	// gradient method
	for(int steps=1; steps<=10000; steps++){
		// check loss
		R loss = calcLoss(G,y,T,theta);
		if(steps == 1 || steps % 50 == 0 || loss < 1e-2){
			std::cout << "step " << steps << ": ";
			std::cout << "label = " << calcLabel(G,T,theta) << "(correct = " << y << "), ";
			std::cout << "loss = " << loss << std::endl;
		}
		if(loss < 1e-2)break;
		// calculate gradient
		Param gradient = calcGradient(G,y,T,theta,epsilon);
		// update parameter
		theta -= alpha * gradient;
	}
	return 0;
}
