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
	
	
	//start of attributes, attribute number, classification location
	dp.setDataInfo(1, 9, 10); // locations for cancer data set
	//dp.setDataInfo(0, 4, 4); //locations for iris data set
	dp.readFile("../Input/Breast_Cancer/breast-cancer-wisconsin.data"); // cancer data set
	//dp.readFile("../Input/Iris/iris.data"); //iris data set
	dp.printData();
	std::string s = "Cancer"; //add tag to log
	//std::string s = "Iris";
	NeuralNet nn = NeuralNet(1000, 0.8f, 0.1, 0.2, 0, true);
	nn.addData(dp.getData());
	nn.addClasses(dp.getClasses());
	nn.storeDatasetName(s);

	//specifies the number of hidden layers and nodes. 15 is the lowest I managed to get to work with a 0.15 learning rate
	int hidden[] = { 12 }; //set the number of hidden nodes in each hidden layer
	nn.initANN(dp.getNumAttributes(), hidden, std::end(hidden) - std::begin(hidden), dp.getClasses().size());
	nn.trainANN();
	nn.useANN();
	
}