/*To read a CSV file with the "shark-ML" library we have to include corresponding headers*/
#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
using namespace shark; 

/*We can use the "importCSV()" method of the "ClassificationDataset" object to load the CSV data from a file.*/
ClassificationDataset dataset; 
importCSV(dataset, "iris_fix.csv", LAST_COLUM); 	

/*Then, we can use this object in ML alogrithms provided by the "Shark-ML" library. 
 * Also, we can also print some statistics about the imported dataset*/
std::size_t classes = numberOfClasses(dataset); 
std::cout << "Number of class" << classes << std::endl; 
std::vector<std::size_t> sizes = classSizes(dataset); 
std::cout << "Class size: " << std::endl; 
for (auto cs : sizes)
{
	std::cout << cs << std::endl; 
}

std::size_t dim = inputDimension(dataset); 
std::cout << "Input dimension" << dim << std::endl; 

