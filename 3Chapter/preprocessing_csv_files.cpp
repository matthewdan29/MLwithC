/*Below, we replace strings with distinct numbersm, but in general, such an approach is a bad idea, for classification tasks. 
 * ML algorithms usually learn only numerical relations, so a more suitale approach would be to use specialized encoding, like one-hot encoding.*/
#include <fstream>
#include <regex>
...
std::ifstream data_stream("iris.data"); 
std::string data_string((std::istreambuf_iterator<char>(data_stream)), std::istreambuf_iterator<char>()); 
data_string = std::regex_replace(data_string, std::regex("Iris-setosa"), "1"); 
data_string = std::regex_replace(data_string, std::regex("Iris-versicolor"), "2"); 
data_string = std::regex_replace(data_string, std::regex("Iris-virginica"), "3"); 
std::ofstream out_stream("iris_fx.csv"); 
out_stream << data_string; 


