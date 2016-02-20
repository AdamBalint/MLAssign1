#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
class DataParser
{
public:
	DataParser();
	~DataParser();
	void readFile(std::string);
	void printData();
	void setAttributeSize(int);
	void setAttributeStart(int);
	void setClassificationLocation(int);
	void setDataInfo(int, int, int);
	std::vector<std::vector<float>> getData();
	std::vector<std::string> getClasses();

private:
	int classExists(std::string);

	std::vector<std::vector<float>> data;
	std::vector<std::string> classes;
	int classLoc = -1;
	int attrStart = -1;
	int attrSize = -1;

};

