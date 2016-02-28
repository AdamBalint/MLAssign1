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
	NeuralNet(int, int, float);//epochs, value of k to use for training, learning rate
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
	void addData(std::vector<std::vector<float>>);
	void addClasses(std::vector<std::string>);
	int getHighest();

private:
	void trainHoldout();
	void trainCrossValidation();
	float getSquaredError(int);
	float average(std::vector<float>);

	std::vector<Node> input; // holds the input nodes
	std::vector<std::vector<Node>> hidden; //hold the hidden layer nodes
	std::vector<Node> output; //hold the output nodes
	int numEpochs; //defines maximum number of epochs
	std::vector<std::vector<float>> trainingSet; // holds the training set
	std::vector<std::vector<float>> testingSet; // holds the testing set
	std::vector<std::vector<float>> dataSet; //holds all the data
	int dataType; // 0 - percentage, 1 - kfold
	float trainingDataPercent;
	int kSize = 0; // number of divisions
	int numInSet; // number of pieces of data in each set
	std::vector<std::string> classes;
	int numSetsTU; //sets how many training sets to use
	double learnRate; // sets the learning rate
	int ansLoc;

};
