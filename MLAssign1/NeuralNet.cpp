#include "NeuralNet.h"


NeuralNet::NeuralNet()
{
	srand(time(NULL)); //set a random seed
}

NeuralNet::NeuralNet(int epochs, int trainToUse, float lr) 
	: numEpochs(epochs), numSetsTU(trainToUse), learnRate(lr)
{
	srand(time(NULL)); //set a random seed
}

NeuralNet::NeuralNet(int epochs, float trainToUse, float lr)
	: numEpochs(epochs), learnRate(lr)
{
	srand(time(NULL)); //set a random seed
	numSetsTU = floor(trainingSet.size()*trainToUse);
}

NeuralNet::~NeuralNet()
{
}

void NeuralNet::addTrainingData(std::vector<std::vector<float>> data){
	trainingSet = data;
	ansLoc = trainingSet.at(0).size()-1;
}

void NeuralNet::addClasses(std::vector<std::string> classes){
	(*this).classes = classes;
}

//initializes network: takes in the architecture of the network
void NeuralNet::initANN(int input, int* hidden, int numHiddenLayers, int output){
	//asks for user to enter learning rate
//	printf("Please enter a learning rate (as decimal): ");
//	std::cin >> learnRate;

	//creates the appropriate amount of nodes in each layer
	printf("Input Size Specified: %d\n", input);
	for (int i = 0; i < input; i++){
		Node n(learnRate, 0.1, 1);
		n.name = "In: " + std::to_string(i);
		(*this).input.push_back(n);
	}

	//supports multiple hidden layers
	printf("Number of hidden Layers: %d\n", numHiddenLayers);
	for (int layer = 0; layer < numHiddenLayers; layer++){
		std::vector<Node> h;
		(*this).hidden.push_back(h);
		for (int i = 0; i < hidden[layer]; i++){
			Node n(learnRate, 0.1, 1);
			n.name = "h" + std::to_string(layer) + "-" + std::to_string(i);
			(*this).hidden.at(layer).push_back(n);
		}
	}

	for (int i = 0; i < output; i++){
		Node n(learnRate, 0.1, 1);
		n.name = "Out: " + std::to_string(i);
		(*this).output.push_back(n);
	}

	//connects the network and then sets up the weights
	connectNetwork();
	initWeights();
}

//connects the network
void NeuralNet::connectNetwork(){
	//sets the inputs of the hidden nodes to each of the input nodes
	//and sets the output of the input nodes as the hidden nodes
	for (int i = 0; i < input.size(); i++){
		for (int j = 0; j < hidden.at(0).size(); j++){
			input.at(i).addOutput(&hidden.at(0).at(j));
			hidden.at(0).at(j).addInput(&input.at(i));
		}
	}
	//connects hidden to hidden and hidden to output the same way as above
	//does not support no hidden layers
	for (int layer = 0; layer < hidden.size(); layer++){
		if (layer + 1 < hidden.size()){
			int nLayer = layer + 1;
			for (int i = 0; i < hidden.at(layer).size(); i++){
				for (int j = 0; j < hidden.at(nLayer).size(); j++){
					hidden.at(layer).at(i).addOutput(&hidden.at(nLayer).at(j));
					hidden.at(nLayer).at(j).addInput(&hidden.at(layer).at(i));
				}
			}
		}
		else{
			for (int i = 0; i < hidden.at(layer).size(); i++){
				for (int j = 0; j < output.size(); j++){
					hidden.at(layer).at(i).addOutput(&output.at(j));
					output.at(j).addInput(&hidden.at(layer).at(i));
				}
			}
		}
	}
}

//initializes the weights by calling each node to generate the amount of weights needed
void NeuralNet::initWeights(){
	for (int i = 0; i < input.size(); i++){
		input.at(i).initWeights();
	}
	for (int i = 0; i < hidden.size(); i++){
		for (int j = 0; j < hidden.at(i).size(); j++){
			hidden.at(i).at(j).initWeights();
		}
	}
	for (int i = 0; i < output.size(); i++){
		output.at(i).initWeights();
	}

}


//does the forward pass through the network on a given input
void NeuralNet::runANN(std::vector<float> values){
	for (int i = 0; i < input.size(); i++){
		input.at(i).setValue(values.at(i));
		//initial push doesn't pass the node value through the sigmoid function
		input.at(i).initialPush();
	}
	for (int i = 0; i < hidden.size(); i++){
		for (int j = 0; j < hidden.at(i).size(); j++){
			//forward does use the sigmoid function
			hidden.at(i).at(j).forward();
		}
	}
	for (int i = 0; i < output.size(); i++){
		output.at(i).forward();
	}
}

//loops through the training examples, and does the correction logic
void NeuralNet::trainANN(){

	std::clock_t start = std::clock(); // get timer to check how long it takes to train
	//loop through the number of epochs specified
	for (int epoch = 0; epoch < numEpochs; epoch++){
		if ((epoch + 1) % 500 == 0){ //only print out every 500 epochs for speed
			printf("Epoch: %d/%d\n", epoch + 1, numEpochs);
			//printNetwork();
		}
		//randomly shuffle the training examples
		auto engine = std::default_random_engine{};
		std::shuffle(trainingSet.begin(), trainingSet.end(), engine);

		//set the correct number predicted to 100% (all of the examples that it will use)
		int correctNum = 0;

		//loop through each training example once
		for (int train = 0; train < numSetsTU; train++){

			//do the forward pass
			runANN(trainingSet.at(train));


			bool incorrect = false; // assume the correct result was predicted
			//loop through all output nodes
		//	for (int i = 0; i < output.size(); i++){
				int fired = getHighest();
				double rawRes = output.at(fired).getOutput();//get raw output
				int correctRes = trainingSet.at(train).at(ansLoc); // get correct result

				for (int i = 0; i < output.size(); i++){
					double out = output.at(i).getOutput();
					double err = (i == correctRes ? 1 : 0) - out;
					output.at(i).addError(err);
				}

				if (fired == (int)trainingSet.at(train).at(ansLoc)){
					//increment correct counter
					correctNum++;
				}

				//incorrect = true; //then the prediction was incorrect
				//double err = correctRes - rawRes; //calculate the error
				//output.at(i).addError(err); //add the error to the output node
				
	//		}
			
			backPass();//and do the back propogation
			adjustWeights();
			
			resetValues(); //reset the value, output and error at all nodes to reset network
		}
		//printf("Correct: %d/%d\n", correctNum, global.numSetsTU); //used to print network for debugging
//		if (correctNum >= numSetsTU * 0.9){ // if all predicted then break
//			printf("All test cases predicted!\n");
//			break;
//		}
		if ((epoch + 1) % 500 == 0){
			printf("Correct: %d/%d\n", correctNum, numSetsTU);
		}

	}
	//print out how long it took to train
	printf("Time to train: %f\n", ((std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)));
}


void NeuralNet::useANN(){
	//ask print out stats
	std::ofstream myfile;

	time_t rawtime = std::time(nullptr);
	struct tm b;
	time(&rawtime);
	b = *localtime(&rawtime);
	char buff[80];
	strftime(buff, 80, "%y_%m_%d_%H_%M_%S", &b);
	

	/*time_t rawTime;
	tm timeInf;
	errno_t res = localtime_s(&timeInf, &rawTime);
	char buff[80];
	asctime_s(buff, 80, &timeInf);*/
	std::string s(buff);
	myfile.open("../Results/Results-" + s + ".txt");
	myfile << "Data Set: " << "TMP" << "\n";
	myfile << "Training Set Size: " << numSetsTU << "\n";
	myfile << "Learning Rate: " << learnRate << "\n";
	myfile << "Momentum: " << "" << "\n";
	myfile << "Learning Type: " << "" << "\n";
	myfile << "Expected\tResult\tAccuracy\tRawResult\n";
	printf("\nLearning rate: %f\n", learnRate);
	printf("Network Type: %d-%d-%d\n\n", input.size(), hidden.at(0).size(), output.size());
	printf("Results for inputs:\n");
	//go through and print out all training examples and info
	for (std::vector<float> inp : trainingSet){
		runANN(inp);
		for (int i = 0; i < trainingSet.at(0).size() - 1; i++){
			printf("%f ", inp.at(i));
		}
		printf("\t");

		int result = getHighest();
		if (result == -1){
			printf("undecided\n");
			result = 0;
		}
		double rawRes = output.at(result).getOutput();
		

		//printf("Expected: %d    Result: %s    %f%% accuracy\n", inp.at(ansLoc), (classes.at(result)).c_str(), ((rawRes)*100.0));
		
		myfile << inp.at(ansLoc) << "\t" << classes.at(result) << "\t" << (rawRes * 100) << "\t" << (rawRes * 100) << "\n";
		std::cout << "Expected: " << inp.at(ansLoc) << "\tResult : " << classes.at(result) << "\t" << (rawRes * 100) << "% accuracy\n";
		printf("Raw result: %f\n", rawRes);
		resetValues();
	}
	myfile.close();
	//allow user to experiment with inputs
	while (true){
		printf("Enter q at any time to quit!\nEnter the 4 bits\n");
		printf("\n\n");
		std::string input;
		std::cin >> input;

		//check if the user wants to quit
		if (input.find('q') != std::string::npos || input.find('Q') != std::string::npos)
			exit(0);
		//check if input is wrong size
		if (input.size() != 4){
			printf("Please enter the correct number of bits!\n");
		}

		//convert string to char array
		char inputArr[5];
		strcpy_s(inputArr, input.c_str());


		std::cmatch cm;
		std::regex reg("([^01]+)");
		//compare regex to input to make sure only 1's and 0's were entered
		if (std::regex_search(inputArr, cm, reg)){
			printf("Please enter valid bits!\n");
		}
		else
		{
			//if okay, then tag extra character to end so algorithm will work
			inputArr[4] = '0';
			input += "0";

			//convert to int array
			std::vector<float> inp;
			for (int i = 0; i < sizeof(inputArr); i++){
				inp.push_back(((int)inputArr[i]) - '0');//-48
			}

			//run the forward pass
			runANN(inp);

			//gather and display results
			int fired = getHighest();
			double rawRes = output.at(fired).getOutput();
			std::cout << "\tResult : " << classes.at(fired) << "\t" << (rawRes * 100) << "% accuracy\n";
			printf("Raw result: %f\n", rawRes);
			//printf("Result is %d with %f%% accuracy\n", fired, (rawRes * 100));
			resetValues(); //reset the network
			
		}

	}

}


//does the backpass, calculates error at each node
void NeuralNet::backPass(){
	for (int i = 0; i < output.size(); i++){
		output.at(i).findError();
	}

	for (int i = hidden.size() - 1; i >= 0; i--){
		for (int j = 0; j < hidden.at(i).size(); j++){
			hidden.at(i).at(j).findError();
		}
	}
}

//adjusts the connection weights
void NeuralNet::adjustWeights(){

	for (int i = 0; i < hidden.size(); i++){
		for (int j = 0; j < hidden.at(i).size(); j++){
			hidden.at(i).at(j).updateWeights();
		}
	}

	for (int i = 0; i < output.size(); i++){
		output.at(i).updateWeights();
	}

}

//resets the network to use again
void NeuralNet::resetValues(){
	for (int i = 0; i < input.size(); i++){
		input.at(i).resetValues();
	}

	for (int i = 0; i < hidden.size(); i++){
		for (int j = 0; j < hidden.at(i).size(); j++){
			hidden.at(i).at(j).resetValues();
		}
	}

	for (int i = 0; i < output.size(); i++){
		output.at(i).resetValues();
	}

}

//prints the network in a readable form
void NeuralNet::printNetwork(){
	printf("Input Layer Size: %d\n", input.size());
	for (Node n : input){
		n.printConnections();
		printf("\n");
	}

	printf("Hidden Layer Number: %d\n", hidden.size());

	for (std::vector<Node> v : hidden){
		printf("Hidden Layer number of nodes %d: \n", v.size());
		for (Node n : v){
			n.printConnections();
			printf("\n");
		}
	}

	printf("Output Layer size: %d ", output.size());
	for (Node n : output){
		n.printConnections();
		printf("\n");
	}
}

int NeuralNet::getHighest(){
	int firedNode = -1;
	double highestNode = -100;
	for (int i = 0; i < output.size(); i++){
		double rawRes = output.at(i).getOutput();//get raw output
		if (rawRes > highestNode){
			highestNode = rawRes;
			firedNode = i;
		}
	}
	return firedNode;
}