#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>

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
using Param = PFN_intern_2019::GNNParam<R>;

const int N = 10;
const int T = 2;
const int D = 8;

const R alpha = 0.0001;
const R eta = 0.9;
const R epsilon = 0.001;

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

    // calculate log
    const int TRY_NUMS = 20;
    vector trainMaxLossLog(EPOCH_NUMS, 0.0), trainMinLossLog(EPOCH_NUMS, 1e9), trainAvgLossLog(EPOCH_NUMS, 0.0);
    vector trainMaxRateLog(EPOCH_NUMS, 0.0), trainMinRateLog(EPOCH_NUMS, 1e9), trainAvgRateLog(EPOCH_NUMS, 0.0);
    vector testMaxLossLog(EPOCH_NUMS, 0.0), testMinLossLog(EPOCH_NUMS, 1e9), testAvgLossLog(EPOCH_NUMS, 0.0);
    vector testMaxRateLog(EPOCH_NUMS, 0.0), testMinRateLog(EPOCH_NUMS, 1e9), testAvgRateLog(EPOCH_NUMS, 0.0);

    for(int tries=1; tries<=TRY_NUMS; tries++){
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

        // stochastic gradient descent method
        for(int epoch=1; epoch<=EPOCH_NUMS; epoch++){
            // permute datasets randomly
            std::vector<int> indexes(TRAIN_NUMS);
            for(int i=0; i<TRAIN_NUMS; i++){
                indexes[i] = i;
            }
            std::shuffle(indexes.begin(), indexes.end(), engine);
            // batch training
            for(int head=0; head+BATCH_SIZE<=TRAIN_NUMS; head+=BATCH_SIZE){
                Param gradientSum(D);
                #pragma omp parallel for reduction(+:gradientSum)
                for(int i=head; i<head+BATCH_SIZE; i++){
                    int id = indexes[i];
                    gradientSum += calcGradient(trainGraphs[id],trainLabels[id],T,theta,epsilon);
                }
                gradientSum /= (R)BATCH_SIZE;
                theta -= alpha * gradientSum;
            }

            // test
            R averageLoss = 0.0, averageRate = 0.0;
            #pragma omp parallel for reduction(+:averageLoss) reduction(+:averageRate)
            for(int i=0; i<TEST_NUMS; i++){
                averageLoss += calcLoss(testGraphs[i],testLabels[i],T,theta);
                averageRate += calcLabel(testGraphs[i],T,theta)==testLabels[i] ? 1.0 : 0.0;
            }
            averageLoss /= TEST_NUMS;
            averageRate /= TEST_NUMS;
            testMaxLossLog(epoch-1) = std::max(testMaxLossLog(epoch-1), averageLoss);
            testMinLossLog(epoch-1) = std::min(testMinLossLog(epoch-1), averageLoss);
            testAvgLossLog(epoch-1) += averageLoss;
            testMaxRateLog(epoch-1) = std::max(testMaxRateLog(epoch-1), averageRate);
            testMinRateLog(epoch-1) = std::min(testMinRateLog(epoch-1), averageRate);
            testAvgRateLog(epoch-1) += averageRate;

            // test for train data
            averageLoss = 0.0, averageRate = 0.0;
            #pragma omp parallel for reduction(+:averageLoss) reduction(+:averageRate)
            for(int i=0; i<TRAIN_NUMS; i++){
                averageLoss += calcLoss(trainGraphs[i],trainLabels[i],T,theta);
                averageRate += calcLabel(trainGraphs[i],T,theta)==trainLabels[i] ? 1.0 : 0.0;
            }
            averageLoss /= TRAIN_NUMS;
            averageRate /= TRAIN_NUMS;
            trainMaxLossLog(epoch-1) = std::max(trainMaxLossLog(epoch-1), averageLoss);
            trainMinLossLog(epoch-1) = std::min(trainMinLossLog(epoch-1), averageLoss);
            trainAvgLossLog(epoch-1) += averageLoss;
            trainMaxRateLog(epoch-1) = std::max(trainMaxRateLog(epoch-1), averageRate);
            trainMinRateLog(epoch-1) = std::min(trainMinRateLog(epoch-1), averageRate);
            trainAvgRateLog(epoch-1) += averageRate;
        }
    }

    testAvgLossLog /= (R)TRY_NUMS;
    testAvgRateLog /= (R)TRY_NUMS;
    trainAvgLossLog /= (R)TRY_NUMS;
    trainAvgRateLog /= (R)TRY_NUMS;
    // output
    std::cout << "testMaxLossLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << testMaxLossLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "testMinLossLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << testMinLossLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "testAvgLossLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << testAvgLossLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "testMaxRateLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << testMaxRateLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "testMinRateLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << testMinRateLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "testAvgRateLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << testAvgRateLog(i) << ",";
    std::cout << "]" << std::endl;
    
    std::cout << "trainMaxLossLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << trainMaxLossLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "trainMinLossLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << trainMinLossLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "trainAvgLossLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << trainAvgLossLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "trainMaxRateLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << trainMaxRateLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "trainMinRateLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << trainMinRateLog(i) << ",";
    std::cout << "]" << std::endl;
    std::cout << "trainAvgRateLog = [";
    for(int i=0; i<EPOCH_NUMS; i++)std::cout << trainAvgRateLog(i) << ",";
    std::cout << "]" << std::endl;

	return 0;
}
