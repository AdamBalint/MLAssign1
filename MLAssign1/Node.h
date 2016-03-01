#pragma once

#include <vector>
#include <string>
class Node
{

public:
	Node(double);
	Node(double, int);
	Node(double, double);
	Node(double, double, int, bool);
	Node();
	~Node();
	void addInput(Node*);
	void addOutput(Node*);
	void initWeights();
	void summationFunc(Node*, double);
	void updateWeights();
	void printConnections();
	double getValue();
	void forward();
	void setValue(double);
	double sigmoid(double);
	double tanhFunc(double);
	double sigmoidDer(double);
	double tanhDer(double);
	void findError();
	void addError(double);
	void resetValues();
	double getOutput();
	void initialPush();
	void resetSumGradients();
	void sumGradient();

	std::string name = "Test";

private:
	void singleExampleWeightUpdate();
	void rpropWeightUpdate();
	int getSign(double);
	

	std::vector<Node*> inputs; // holds all the inputs to the node
	std::vector<double> weights; //holds all the weights of the inputs
	std::vector<double> prevGradients;
	std::vector<double> deltaWeights;
	std::vector<double> sumGradients;
	std::vector<Node*> outputs; //hold all the nodes that this node outputs to
	
	double learningRate; //hold the learning rate
	const double e = 2.71828182845904523536; // the value of e to use in the sigmoid function
	double value = 0; //the value of this node
	double err = 0; // the error of this node
	double output; // the output of the node (inputs not sigmoided, the rest are)
	double momentum;
	int learningMethod = 0;
	double lastChange = 0;
	bool single = true;
};