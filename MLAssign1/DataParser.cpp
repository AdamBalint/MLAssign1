#include "DataParser.h"


DataParser::DataParser()
{
}


DataParser::~DataParser()
{
}

void DataParser::readFile(std::string file){
	std::ifstream in;
	in.open(file);

	std::string line;
	//loops while there are things in the file
	while (getline(in, line)){
		if (line.length() != 0){ //all inputs must be 5. 4 for the inputs and 1 for the teacher parity bit
			std::vector<std::string> tokens;
			std::string tmp;
			std::stringstream ss(line);
			while (getline (ss, tmp, ',')){
				tokens.push_back(tmp);
			}

			std::vector<float> res;
			for (int i = attrStart; i < attrSize; i++){
				res.push_back(std::atof(tokens.at(i).c_str()));
			}

			int clsLoc = classExists(tokens.at(classLoc));
			if (clsLoc == -1){
				classes.push_back(tokens.at(classLoc));
				res.push_back(classes.end()-classes.begin());
			}
			else {
				res.push_back(clsLoc);
			}
			
			data.push_back(res); //add to collection
		}
	}
	in.close(); //close the file
}

void DataParser::printData(){
	for (std::vector<float> v : data){
		for (float a : v)
			printf("%f, ", a);
		printf("\n");
	}
	printf("Number of Classifications: %d", classes.size());
}

void DataParser::setAttributeStart(int attributeStart){
	attrStart = attributeStart;
}
void DataParser::setAttributeSize(int attributeSize){
	attrSize = attributeSize;
}
void DataParser::setClassificationLocation(int classificationLoc){
	classLoc = classificationLoc;
}

void DataParser::setDataInfo(int attributeStart, int attributeSize, int classificationLoc){
	attrStart = attributeStart;
	attrSize = attributeSize;
	classLoc = classificationLoc;
}

int DataParser::classExists(std::string cls){
	int loc = std::find(classes.begin(), classes.end(), cls) - classes.begin();
	if (loc == classes.size())
		return -1;
	return loc;
}
