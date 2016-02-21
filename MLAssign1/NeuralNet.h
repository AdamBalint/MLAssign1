#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <ctime>
#include <regex>
#include <random>
#include "Node.h"

class NeuralNet
{
public:
	NeuralNet();
	NeuralNet(int, int, float);
	~NeuralNet();
	
	void initANN(int, int*, int, int);
	void connectNetwork();
	void printNetwork();
	void initWeights();
	void runANN(std::vector<float>);
	void trainANN();
	void useANN();
	void backPass();
	void adjustWeights();
	void resetValues();
	void addTrainingData(std::vector<std::vector<float>>);
	int getHighest();

private:
	std::vector<Node> input; // holds the input nodes
	std::vector<std::vector<Node>> hidden; //hold the hidden layer nodes
	std::vector<Node> output; //hold the output nodes
	int numEpochs; //defines maximum number of epochs
	std::vector<std::vector<float>> trainingSet; // first 4 actual bits, 5th is the parity bit
	int numSetsTU; //sets how many training sets to use
	double learnRate; // sets the learning rate
	int ansLoc;

};
