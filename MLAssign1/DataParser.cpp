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
			while (getline(ss, tmp, ',')){
				tokens.push_back(tmp);
			}
				std::vector<float> res;
				for (int i = attrStart; i < attrSize+attrStart; i++){
					res.push_back(std::atof(tokens.at(i).c_str()));
					if (highest.size() < attrSize){
						highest.push_back(res.back());
					} 
					else if (highest.at(i - attrStart) < res.back()){
						highest.at(i - attrStart) = res.back();
					}
				}

				int clsLoc = classExists(tokens.at(classLoc));
				if (clsLoc == -1){
					classes.push_back(tokens.at(classLoc));
					res.push_back(classes.size()-1);
				}
				else {
					res.push_back(clsLoc);
				}

				data.push_back(res); //add to collection
		}
	}
	in.close(); //close the file
	printf("Data Size: %d\tClasses: %d\n", data.size(), classes.size());
	normalize();
	printf("Normalized Data");
}

void DataParser::normalize(){
	for (int i = 0; i < data.size(); i++){
		for (int j = 0; j < data.at(i).size()-1; j++){
			data.at(i).at(j) /= highest.at(j);
		}
	}
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

std::vector<std::vector<float>> DataParser::getData(){
	return data;
}

std::vector<std::string> DataParser::getClasses(){
	return classes;
}

int DataParser::getNumAttributes(){
	return attrSize;
}