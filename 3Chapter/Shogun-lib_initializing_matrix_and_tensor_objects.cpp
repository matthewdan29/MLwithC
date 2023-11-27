/*Using the "Shogun" library we can use a particular constructor of the "SGMatrix" type to initialize it with the C++ array. 
 * It takes a pointer to the data and matrix dimensions*/
std::vector<double> values; 
...
SGMatrix<float64_t> matrix(values.data(), num_rows, numcols); 


