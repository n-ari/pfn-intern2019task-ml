#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../common/structures.hpp"
#include "../common/functions.hpp"
#include "../common/gnn.hpp"

using R = double;

using vector = PFN_intern_2019::vector<R>;
using matrix = PFN_intern_2019::matrix<R>;
using graph = PFN_intern_2019::graph<R>;
using PFN_intern_2019::calcLoss;
using PFN_intern_2019::calcLabel;
using PFN_intern_2019::calcGradient;
using PFN_intern_2019::elDiv;
using Param = PFN_intern_2019::GNNParam<R>;

const int N = 10;
const int T = 2;
const int D = 8;

const R alpha = 0.001;
const R beta1 = 0.9;
const R beta2 = 0.999;
const R epsilon = 1e-8;

const R gradEpsilon = 0.001;

const int TRAIN_NUMS = 1500;	// 2000 * 0.75
const int TEST_NUMS = 500;		// 2000 * 0.25

const int EPOCH_NUMS = 200;
const int BATCH_SIZE = 10;
const int BATCH_NUMS = TRAIN_NUMS / BATCH_SIZE;

#ifdef _OPENMP
#include <omp.h>
#pragma omp declare reduction(+:Param:omp_out += omp_in)
#endif

// load graph from file subroutine
// does NOT handle exceptions
graph loadGraphFromFile(const std::string &filename){
	std::ifstream ifs(filename);
	int N;
	ifs >> N;
	graph G(N);
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			int val;
			ifs >> val;
			if(val == 1){
				G.addEdge(i,j);
			}
		}
	}
	ifs.close();
	return G;
}

// load label from file subroutine
// does NOT handle exceptions
bool loadLabelFromFile(const std::string &filename){
	std::ifstream ifs(filename);
	int label;
	ifs >> label;
	ifs.close();
	return label == 1;
}

int main(){
	// load training data sets
	std::vector<graph> trainGraphs;
	std::vector<bool> trainLabels;
	{
		trainGraphs.reserve(TRAIN_NUMS);
		trainLabels.reserve(TRAIN_NUMS);
		for(int i=0; i<TRAIN_NUMS; i++){
			trainGraphs.push_back(loadGraphFromFile("../../datasets/train/" + std::to_string(i) + "_graph.txt"));
			trainLabels.push_back(loadLabelFromFile("../../datasets/train/" + std::to_string(i) + "_label.txt"));
		}
	}
	// load test data sets
	std::vector<graph> testGraphs;
	std::vector<bool> testLabels;
	{
		testGraphs.reserve(TEST_NUMS);
		testLabels.reserve(TEST_NUMS);
		for(int i=0; i<TEST_NUMS; i++){
			testGraphs.push_back(loadGraphFromFile("../../datasets/train/" + std::to_string(TRAIN_NUMS + i) + "_graph.txt"));
			testLabels.push_back(loadLabelFromFile("../../datasets/train/" + std::to_string(TRAIN_NUMS + i) + "_label.txt"));
		}
	}
	// RNG
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());

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

	// adam
	Param m(D), v(D);
	int t = 0;
	for(int epoch=1; epoch<=EPOCH_NUMS; epoch++){
		// permute datasets randomly
		std::vector<int> indexes(TRAIN_NUMS);
		for(int i=0; i<TRAIN_NUMS; i++){
			indexes[i] = i;
		}
		std::shuffle(indexes.begin(), indexes.end(), engine);
		// batch training
		for(int head=0; head+BATCH_SIZE<=TRAIN_NUMS; head+=BATCH_SIZE){
			t = t + 1;
			Param gradientSum(D);
			#pragma omp parallel for reduction(+:gradientSum)
			for(int i=head; i<head+BATCH_SIZE; i++){
				int id = indexes[i];
				gradientSum += calcGradient(trainGraphs[id],trainLabels[id],T,theta,gradEpsilon);
			}
			gradientSum /= (R)BATCH_SIZE;
			
			m = beta1 * m + (1.0-beta1) * gradientSum;
			v = beta2 * v + (1.0-beta2) * gradientSum.map([](R x){return x*x;});
			Param mhat = m / (1.0-std::pow(beta1,t));
			Param vhat = v / (1.0-std::pow(beta2,t));
			theta = theta - elDiv(alpha * mhat, vhat.map([](R x){return std::sqrt(x);}) + epsilon);
		}

		// test
		if(epoch==1 || epoch%10==0 || epoch==EPOCH_NUMS){
			R averageLoss = 0.0, averageRate = 0.0;
			#pragma omp parallel for reduction(+:averageLoss) reduction(+:averageRate)
			for(int i=0; i<TEST_NUMS; i++){
				averageLoss += calcLoss(testGraphs[i],testLabels[i],T,theta);
				averageRate += calcLabel(testGraphs[i],T,theta)==testLabels[i] ? 1.0 : 0.0;
			}
			averageLoss /= TEST_NUMS;
			averageRate /= TEST_NUMS;
			std::cout << "epoch(" << epoch << "): ";
			std::cout << "test avg loss = " << std::fixed << std::setprecision(5) << averageLoss << ", ";
			std::cout << "test avg rate = " << std::fixed << std::setprecision(5) << averageRate << " / ";

			// (train set)
			averageLoss = 0.0, averageRate = 0.0;
			for(int i=0; i<TRAIN_NUMS; i++){
				averageLoss += calcLoss(trainGraphs[i],trainLabels[i],T,theta);
				averageRate += calcLabel(trainGraphs[i],T,theta)==trainLabels[i] ? 1.0 : 0.0;
			}
			averageLoss /= TRAIN_NUMS;
			averageRate /= TRAIN_NUMS;
			std::cout << "train avg loss = " << std::fixed << std::setprecision(5) << averageLoss << ", ";
			std::cout << "train avg rate = " << std::fixed << std::setprecision(5) << averageRate << std::endl;
		}
	}

	return 0;
}
