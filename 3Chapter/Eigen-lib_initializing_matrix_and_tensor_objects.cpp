/*Using the "Eigen" library we can wrap a C++ array into the "Eigen::Matrix" object with the "Eigen::Map" type. 
 * The wrapped object will behave as a standard "Eigen" matrix. 
 * We have to parametrize the "Eigen::Map" type with the type of matrix that has the required behavior. 
 * Also, when we create the "Eigen::Map" object, it takes as arguments a pointer to the C++ array and matrix dimensions*/
std::vector<double> values; 
...
auto x_data = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows_num, columns_num); 

