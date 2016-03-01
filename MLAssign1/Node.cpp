#include "Node.h"

//init node with learning rate that is specified
Node::Node(double learningRate)
	: learningRate(learningRate)
{
}

Node::Node(double learningRate, int learningMethod)
	: learningRate(learningRate), learningMethod(learningMethod)
{
}

Node::Node(double learningRate, double momentum)
	: learningRate(learningRate), momentum(momentum)
{
}

Node::Node(double learningRate, double momentumm, int learningMethod, bool single)
	: learningRate(learningRate), momentum(momentum), learningMethod(learningMethod), single(single)
{
}

//init node with default learning rate
Node::Node()
	: learningRate(0.2){
}

Node::~Node()
{
}
//adds input to node
void Node::addInput(Node *in){
	inputs.push_back(in);
}

//adds output to node
void Node::addOutput(Node *in){
	outputs.push_back(in);
}

//initializes the weights to between -0.5 and 0.5. if really close to 0, then it makes it not as close
void Node::initWeights(){
	for (int i = 0; i < inputs.size(); i++){
		double weight = ((double)rand() / ((double)RAND_MAX + 1) - 0.5);
		printf("Weight: %f\n", weight);
		if (abs(weight) < 0.001)
			weight *= 10;
		weights.push_back(weight);
		if (!single){
			prevGradients.push_back(0.1);
			deltaWeights.push_back(0.1);
			sumGradients.push_back(0);
		}

	}
}

//updates the weights using the formula provided in the BP example slides
void Node::updateWeights(){
	if (single)
		singleExampleWeightUpdate();
	else
		rpropWeightUpdate();

}

void Node::singleExampleWeightUpdate(){
	for (int i = 0; i < inputs.size(); i++){
			double derValue = 0;
			if (learningMethod == 0)
				derValue = sigmoidDer(value);
			else if (learningMethod == 1)
				derValue = tanhDer(value);
			double errAtNode = err * derValue;
			double adjust = learningRate * errAtNode * ((*inputs.at(i)).getOutput());
			double mom = momentum * lastChange;
			lastChange = adjust + mom;
			double nWeight = weights.at(i) + adjust + mom;
			weights.at(i) = nWeight;
	}
}

void Node::rpropWeightUpdate(){
	double npos = 1.2, nneg = 0.55;
	double deltamax = 50, deltamin = 0.00000001;
	double derValue = 0;
	if (learningMethod == 0)
		derValue = sigmoidDer(value);
	else if (learningMethod == 1)
		derValue = tanhDer(value);

	for (int i = 0; i < inputs.size(); i++){
		
		double gradient = sumGradients.at(i);
		double delta;
		bool skipUpdate = false;
		switch (getSign(gradient * prevGradients.at(i))){
		case 1:
			
			delta = deltaWeights.at(i) * getSign(gradient);
			deltaWeights.at(i) = fmin(deltaWeights.at(i) * npos, deltamax);
//			printf("increase delta\n");
			break;
		case -1:
			deltaWeights.at(i) = fmax(deltaWeights.at(i) * nneg, deltamin);
			gradient = 0;
			skipUpdate = true;
//			printf("decrease delta\n");
			break;
		case 0:
//			printf("nothing\n");
			delta = deltaWeights.at(i) * getSign(gradient);
			break;
		}
		if (!skipUpdate){
			weights.at(i) = weights.at(i) - delta;
			prevGradients.at(i) = gradient;
		}
	}
}

int Node::getSign(double in){
	if (in < 0)
		return -1;
	else if (in > 0)
		return 1;
	
	return 0;
}

void Node::sumGradient(){
	for (int i = 0; i < inputs.size(); i++){
		double derValue = 0;
		if (learningMethod == 0)
			derValue = sigmoidDer(value);
		else if (learningMethod == 1)
			derValue = tanhDer(value);
		
		double gradient = -err * derValue * ((*inputs.at(i)).getOutput());
		sumGradients.at(i) += gradient;
	}
}

//finds the error contribution of all input nodes, and adds to running total in input
void Node::findError(){
	for (int i = 0; i < inputs.size(); i++){
		double pError = weights.at(i) * err;
		inputs.at(i)->addError(pError);
	}
}

//adds the error contribution of this node do a running total
void Node::addError(double error){
	err += error;
}

//sums up all the input * weight values and adds to running total
void Node::summationFunc(Node *n, double val){
	bool notFound = true;
	for (int i = 0; i < inputs.size(); i++){
		if (inputs.at(i) == n){ //checks to find which weight to use
			value += weights.at(i) * val;
			notFound = false;
			break;
		}
	}
	if (notFound) //if weight happens not to be found then it prints message
		printf("Node for summation not found!!!!\n");
}


//prints the connections of the node in a readable form
void Node::printConnections(){
	printf("Name: %s\n", name.c_str());
	printf("Contains %d input connections: \n", inputs.size());
	for (int i = 0; i < inputs.size(); i++){
		printf("In: %s \t", (*inputs.at(i)).name.c_str());
		printf("Weight: %f: \n", weights.at(i));
	}
	printf("\nContains %d output connections: \n", outputs.size());
	for (int i = 0; i < outputs.size(); i++){
		printf("In: %s \t", (*outputs.at(i)).name.c_str());
		//		printf("Weight: %f: \n", weights.at(i));
	}
	printf("\n");
}

//gets the value of the node
double Node::getValue(){
	return value;
}

//pushes the value through the sigmoid and passes it to the nodes in this node's output
//passes a pointer to itself so that the weight can be found
void Node::forward(){
	if (learningMethod == 0)
		output = sigmoid(value);
	else if (learningMethod == 1)
		output = tanhFunc(value);
	//else
		//put in rprop here
	for (int i = 0; i < outputs.size(); i++){
		outputs.at(i)->summationFunc(this, output);
	}
}
//same as above, but used for the input nodes, so no sigmoid is used
void Node::initialPush(){
	output = value;
	for (int i = 0; i < outputs.size(); i++){
		outputs.at(i)->summationFunc(this, output);
	}
}

//defines the sigmoid function
double Node::sigmoid(double in){
	return (1.0 / (1.0 + pow(e, -in)));
}

//defines the sigmoid derivative
double Node::sigmoidDer(double in){
	return sigmoid(in)*(1.0 - sigmoid(in));
}

double Node::tanhFunc(double in){
	return tanh(in);
}

double Node::tanhDer(double in){
	return 1 - (tanh(in)*tanh(in));
}

//sets the value of the node. Used to set the inputs
void Node::setValue(double inputVal){
	value = inputVal;
}

//gets the output of the node
double Node::getOutput(){
	return output;
}

//resets the node to be used again
void Node::resetValues(){
		value = 0;
		err = 0;
	output = 0;
}
void Node::resetSumGradients(){
	std::vector<double> n;
	for (int i = 0; i < inputs.size(); i++){
		n.push_back(0);
	}
	sumGradients = n;
}