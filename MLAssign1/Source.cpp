#include <iostream>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <regex>
#include <random>
#include "Node.h"
#include "DataParser.h"
#include "NeuralNet.h"


int main(int argc, char* argv[]){
	
	DataParser dp;
	dp.setDataInfo(1, 9, 10);

	//dp.readFile("../Input/input.txt");
	//dp.readFile("../Input/Breast_Cancer/wdbc.data");
	dp.readFile("../Input/Breast_Cancer/breast-cancer-wisconsin.data");
	//dp.readFile("../Input/Iris/iris.data");
	dp.printData();

	NeuralNet nn = NeuralNet(1500, 0.80f, 0.01, 0.2, false);
	nn.addData(dp.getData());
	nn.addClasses(dp.getClasses());
	//specifies the number of hidden layers and nodes. 15 is the lowest I managed to get to work with a 0.15 learning rate
	int hidden[] = { 22 }; //set the number of hidden nodes in each hidden layer
	nn.initANN(dp.getNumAttributes(), hidden, std::end(hidden) - std::begin(hidden), 2);
	nn.trainANN();
	nn.useANN();
	//initANN(4, hidden, std::end(hidden) - std::begin(hidden), 1); //init the neural net
	//trainANN(); //train the neural net
	//useANN(); //allow user to use net
	//int a;
	//std::cin >> a;
}