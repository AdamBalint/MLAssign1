#include "NeuralNet.h"


NeuralNet::NeuralNet()
{
	srand(time(NULL)); //set a random seed
	time_t rawtime = std::time(nullptr);
	struct tm b;
	time(&rawtime);
	b = *localtime(&rawtime);
	char buff[80];
	strftime(buff, 80, "%y_%m_%d_%H_%M_%S", &b);
	timestamp = std::string(buff);
}

NeuralNet::NeuralNet(int epochs, int kValue, float lr, float momentum, int activationFunc, bool single) 
	: numEpochs(epochs), kSize(kValue), learnRate(lr), momentum(momentum), learningMethod(activationFunc), single(single)
{
	srand(time(NULL)); //set a random seed
	dataType = 1;
	time_t rawtime = std::time(nullptr);
	struct tm b;
	time(&rawtime);
	b = *localtime(&rawtime);
	char buff[80];
	strftime(buff, 80, "%y_%m_%d_%H_%M_%S", &b);
	timestamp = std::string(buff);
}

NeuralNet::NeuralNet(int epochs, float trainToUse, float lr, float momentum, int activationFunc, bool single)
	: numEpochs(epochs), trainingDataPercent(trainToUse), learnRate(lr), momentum(momentum), learningMethod(activationFunc), single(single)
{
	srand(time(NULL)); //set a random seed
	dataType = 0;
	time_t rawtime = std::time(nullptr);
	struct tm b;
	time(&rawtime);
	b = *localtime(&rawtime);
	char buff[80];
	strftime(buff, 80, "%y_%m_%d_%H_%M_%S", &b);
	timestamp = std::string(buff);
}

NeuralNet::~NeuralNet()
{
}

void NeuralNet::storeDatasetName(std::string name){
	dataSetName = name;
}

void NeuralNet::addData(std::vector<std::vector<float>> data){
	ansLoc = data.at(0).size() - 1;
	dataSet = data;
	numSetsTU = floor(data.size()*trainingDataPercent);
	auto engine = std::default_random_engine{};
	std::shuffle(dataSet.begin(), dataSet.end(), engine);
	if (dataType == 0){
		std::vector<std::vector<float>>::const_iterator start = data.begin();
		std::vector<std::vector<float>>::const_iterator end = data.begin() + numSetsTU;
		trainingSet = std::vector<std::vector<float>>(start, end);
		std::vector<std::vector<float>>::const_iterator dataEnd = data.end();
		testingSet = std::vector<std::vector<float>>(end, dataEnd);
	}
	else{
		std::vector<std::vector<float>>::const_iterator start = data.begin();
		std::vector<std::vector<float>>::const_iterator end = data.end();
		trainingSet = std::vector<std::vector<float>>(start, end);
		numInSet = trainingSet.size() / kSize;
	}


}

void NeuralNet::addClasses(std::vector<std::string> classes){
	(*this).classes = classes;
}

//initializes network: takes in the architecture of the network
void NeuralNet::initANN(int input, int* hidden, int numHiddenLayers, int output){

	//creates the appropriate amount of nodes in each layer
	printf("Input Size Specified: %d\n", input);
	for (int i = 0; i < input; i++){
		Node n(learnRate, momentum, learningMethod, single);
		n.name = "In: " + std::to_string(i);
		(*this).input.push_back(n);
	}

	//supports multiple hidden layers
	printf("Number of hidden Layers: %d\n", numHiddenLayers);
	for (int layer = 0; layer < numHiddenLayers; layer++){
		std::vector<Node> h;
		(*this).hidden.push_back(h);
		for (int i = 0; i < hidden[layer]; i++){
			Node n(learnRate, momentum, learningMethod, single);
			n.name = "h" + std::to_string(layer) + "-" + std::to_string(i);
			(*this).hidden.at(layer).push_back(n);
		}
	}

	for (int i = 0; i < output; i++){
		Node n(learnRate, momentum, learningMethod, single);
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
	if (dataType == 0)
		trainHoldout();
	else if (dataType == 1)
		trainCrossValidation();
}

void NeuralNet::trainHoldout(){
	std::ofstream myfile;
	myfile.open("../Results/EpochSummary-" + timestamp + ".txt");
	myfile << "Epoch\tsum error squared(train)\tsum error squared(test)\tavg error squared(train)\tavg error squared(test)\tpercent correct(train)\tpercent correct(test)\n";

	std::clock_t start = std::clock(); // get timer to check how long it takes to train
	//loop through the number of epochs specified
	for (int epoch = 0; epoch < numEpochs; epoch++){
		printf("Epoch: %d/%d\n", epoch + 1, numEpochs);

		//randomly shuffle the training examples
		auto engine = std::default_random_engine{};
		std::shuffle(trainingSet.begin(), trainingSet.end(), engine);

		//set the correct number predicted to 100% (all of the examples that it will use)

		int correctNumTrain = 0;
		int correctNumTest = 0;
		double squareErrorTrain = 0;
		int numInSquareErrorTrain = 0;
		double squareErrorTest = 0;
		int numInSquareErrorTest = 0;

		//loop through each training example once
		for (int train = 0; train < trainingSet.size(); train++){
			numInSquareErrorTrain++;
			//do the forward pass
			runANN(trainingSet.at(train));


			bool incorrect = false; // assume the correct result was predicted
			int fired = getHighest();
			double rawRes = output.at(fired).getOutput();//get raw output
			int correctRes = trainingSet.at(train).at(ansLoc); // get correct result

			for (int i = 0; i < output.size(); i++){
				double out = output.at(i).getOutput();
				double err = (i == correctRes ? 1 : 0) - out;
				output.at(i).addError(err);
				squareErrorTrain += pow(err, 2);
			}

			if (fired == (int)trainingSet.at(train).at(ansLoc)){
				//increment correct counter
				correctNumTrain++;
			}


			backPass();//and do the back propogation
			if (single){
				adjustWeights();
			}
			else{
				findGradients();
			}

			resetValues(); //reset the value, output and error at all nodes to reset network
		}


		if (!single){
			adjustWeights();
			resetGradients();
		}
		
			for (int test = 0; test < testingSet.size(); test++){
				numInSquareErrorTest++;
				runANN(testingSet.at(test));

				bool incorrect = false; // assume the correct result was predicted
				int fired = getHighest();
				double rawRes = output.at(fired).getOutput();//get raw output
				int correctRes = testingSet.at(test).at(ansLoc); // get correct result

				for (int i = 0; i < output.size(); i++){
					double out = output.at(i).getOutput();
					double err = (i == correctRes ? 1 : 0) - out;
					output.at(i).addError(err);
					squareErrorTest += pow(err, 2);
				}

				if (fired == (int)testingSet.at(test).at(ansLoc)){
					//increment correct counter
					correctNumTest++;
				}
				resetValues(); //reset the value, output and error at all nodes to reset network
			}
			printf("Correct: Train: %d/%d\tTest: %d/%d\n", correctNumTrain, numInSquareErrorTrain, correctNumTest, numInSquareErrorTest);
			
			//**************************log data for each fold here************************************
			myfile << epoch << "-" << epoch << "\t" << squareErrorTrain << "\t" << squareErrorTest << "\t" << (squareErrorTrain / numInSquareErrorTrain);
			myfile << "\t" << (squareErrorTest / numInSquareErrorTest) << "\t" << (((double)correctNumTrain) / numInSquareErrorTrain) << "\t" << (((double)correctNumTest) / numInSquareErrorTest) << "\n";

	}
	//print out how long it took to train
	printf("Time to train: %f\n", ((std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)));
	myfile.flush();
	myfile.close();
}

void NeuralNet::trainCrossValidation(){
	std::ofstream myfile;
	myfile.open("../Results/EpochSummary-" + timestamp + ".txt");
	myfile << "Epoch\tsum error squared(train)\tsum error squared(test)\tavg error squared(train)\tavg error squared(test)\tpercent correct(train)\tpercent correct(test)\n";


	std::clock_t start = std::clock(); // get timer to check how long it takes to train
	//loop through the number of epochs specified
	std::vector<float> oldRes;
	float oldResult = 100000;
	int noImprovCount = 0;
	for (int epoch = 0; epoch < numEpochs; epoch++){
		printf("Epoch %d\n", epoch);

		auto engine = std::default_random_engine{};
		std::shuffle(trainingSet.begin(), trainingSet.end(), engine);

		int totalCorrect = 0;
		float newRes = 0;
		for (int k = 0; k < kSize; k++){
			
			printf("\tK: %d: ", k);

			//set the correct number predicted to 100% (all of the examples that it will use)
			int correctNumTrain = 0;
			int correctNumTest = 0;
			double squareErrorTrain = 0;
			int numInSquareErrorTrain = 0;
			double squareErrorTest = 0;
			int numInSquareErrorTest = 0;

			//loop through each training example once
			for (int train = 0; train < trainingSet.size(); train++){
				bool inTest = false;
				if (k == kSize - 1){
					if (train >= k*numInSet)
						inTest = true;
				}
				else
					if (train >= k*numInSet && train < ((k + 1) % kSize)*numInSet)
						inTest = true;

				//do the forward pass
				runANN(trainingSet.at(train));


				bool incorrect = false; // assume the correct result was predicted
				//loop through all output nodes
				int fired = getHighest();
				double rawRes = output.at(fired).getOutput();//get raw output
				int correctRes = trainingSet.at(train).at(ansLoc); // get correct result

				for (int i = 0; i < output.size(); i++){
					double out = output.at(i).getOutput();
					double err = (i == correctRes ? 1 : 0) - out;
					output.at(i).addError(err);
					if (inTest)
						squareErrorTest += pow(err, 2);
					else
						squareErrorTrain += pow(err, 2);
				}

				if (fired == (int)trainingSet.at(train).at(ansLoc)){
					if (inTest)
						correctNumTest++;
					else
						correctNumTrain++;
				}

				if (!inTest){
					numInSquareErrorTrain++;
					backPass();//and do the back propogation
					if (single){
						adjustWeights();
					}
					else{
						findGradients();
					}
				}
				else{
					numInSquareErrorTest++;
				}
				


				resetValues(); //reset the value, output and error at all nodes to reset network
			}
			newRes += squareErrorTrain / numInSquareErrorTrain;
			//totalCorrect += correctNum;
			printf("Correct: Train: %d/%d\tTest: %d/%d\n", correctNumTrain, numInSquareErrorTrain, correctNumTest, numInSquareErrorTest);
			if (!single){
				adjustWeights();
				resetGradients();
			}

			//**************************log data for each fold here************************************
			myfile << epoch << "-" << k << "\t" << squareErrorTrain << "\t" << squareErrorTest << "\t" << (squareErrorTrain / numInSquareErrorTrain);
			myfile << "\t" << (squareErrorTest / numInSquareErrorTest) << "\t" << (((double)correctNumTrain) / numInSquareErrorTrain) << "\t" << (((double)correctNumTest) / numInSquareErrorTest) <<"\n";
		}
		

			if (oldResult - newRes > 0){
				printf("Change from error: %f\n\n", (oldResult - newRes));
				oldResult = newRes;
				noImprovCount = 0;
			}
			else{
				printf("Change from error: -%f\n\n", (oldResult - newRes));
				noImprovCount++;
			}
	}
	myfile.flush();
	myfile.close();
		
}

float NeuralNet::average(std::vector<float> in){
	float sum = 0;
	for (int i = 0; i < in.size(); i++){
		sum += in.at(i);
	}
	return sum / in.size();
}


void NeuralNet::useANN(){
	//ask print out stats
	std::ofstream myfile;

	printf("\nLearning rate: %f\n", learnRate);
	printf("Network Type: %d-%d-%d\n\n", input.size(), hidden.at(0).size(), output.size());
	printf("Results for inputs:\n");


	myfile.open("../Results/FinalClassification-" + timestamp + ".txt");
	myfile << "Data Set: " << dataSetName << "\n";
	bool kfld = trainingSet.size() == 0;
	myfile << "Data Partitioning: " << (kfld ? "K Fold\n" : "Holdout\n");
	if (!kfld){
		myfile << "Training Set Size: " << trainingSet.size() << "\n";
		myfile << "Testing Set Size: " << testingSet.size() << "\n";
	}
	else
		myfile << "Number of folds: " << kSize << "\n";
	myfile << "Learning Rate: " << learnRate << "\n";
	myfile << "Momentum: " << momentum << "\n";
	myfile << "Activation Function: " << (learningMethod == 0 ? "sigmoid":"tanh") << "\n";
	myfile << "Network: " << input.size() << "-" << hidden.at(0).size() << "-" << output.size() << "\n";
	myfile << "Training type: " << (single ? "Live Training":"Batch Training")<< "\n";
	myfile << "\nExpected\tResult\tAccuracy\tRawResult\n";
	
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
		
		myfile << inp.at(ansLoc) << "\t" << classes.at(result) << "\t" << (rawRes * 100) << "\t" << rawRes << "\tTrain\n";
		std::cout << "Expected: " << inp.at(ansLoc) << "\tResult : " << classes.at(result) << "\t" << (rawRes * 100) << "% accuracy\n";
		printf("Raw result: %f\n", rawRes);
		resetValues();
	}
	for (std::vector<float> inp : testingSet){
		runANN(inp);
		for (int i = 0; i < testingSet.at(0).size() - 1; i++){
			printf("%f ", inp.at(i));
		}
		printf("\t");

		int result = getHighest();
		if (result == -1){
			printf("undecided\n");
			result = 0;
		}
		double rawRes = output.at(result).getOutput();

		myfile << inp.at(ansLoc) << "\t" << classes.at(result) << "\t" << (rawRes * 100) << "\t" << rawRes << "\tTest\n";
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

//adjusts the connection weights
void NeuralNet::findGradients(){
	for (int i = 0; i < hidden.size(); i++){
		for (int j = 0; j < hidden.at(i).size(); j++){
			hidden.at(i).at(j).sumGradient();
		}
	}
	for (int i = 0; i < output.size(); i++){
		output.at(i).sumGradient();
	}
}

void NeuralNet::resetGradients(){
	for (int i = 0; i < hidden.size(); i++){
		for (int j = 0; j < hidden.at(i).size(); j++){
			hidden.at(i).at(j).resetSumGradients();
		}
	}
	for (int i = 0; i < output.size(); i++){
		output.at(i).resetSumGradients();
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