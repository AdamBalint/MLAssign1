#define _CRT_SECURE_NO_WARNINGS
#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <ctime>
#include <regex>
#include <random>
#include "Node.h"
#include <fstream>
#include <ctime>
#include <time.h>
#include <stdio.h>

class NeuralNet
{
public:
	NeuralNet(); // default constructor
	NeuralNet(int, int, float);//epochs, # of data to use for training, learning rate
	NeuralNet(int, float, float);//epochs, % of data to use for training, learning rate

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
	void addClasses(std::vector<std::string>);
	int getHighest();

private:
	std::vector<Node> input; // holds the input nodes
	std::vector<std::vector<Node>> hidden; //hold the hidden layer nodes
	std::vector<Node> output; //hold the output nodes
	int numEpochs; //defines maximum number of epochs
	std::vector<std::vector<float>> trainingSet; // first 4 actual bits, 5th is the parity bit
	std::vector<std::vector<float>> testingSet;
	std::vector<std::vector<float>> dataSet;
	std::vector<std::string> classes;
	int numSetsTU; //sets how many training sets to use
	double learnRate; // sets the learning rate
	int ansLoc;

};
