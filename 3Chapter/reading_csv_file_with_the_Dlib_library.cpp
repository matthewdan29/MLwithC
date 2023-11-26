#include <Dlib/matrix.h>
using namespace Dlib; 

/*Now we define the "matrix" object and load data from the file*/
matrix<double> data; 
std::ifstream file("iris_fix.csv"); 
file >> data; 
std::cout << data << std::endl; 
