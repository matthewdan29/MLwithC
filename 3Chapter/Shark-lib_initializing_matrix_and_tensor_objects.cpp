/*The "Shark-ML" framework has special adaptor function that create wrappers for C++ arrays. 
 * These functions create objects that behave as regular "Shark-ML" matrices. 
 * To wrap a C++ container with adaptor functions, we have to pass a pointer to the data and corresponding dimensions as argumenst.*/
std::vector<float> data{1,2,3,4}; 
auto m = remora::dense_matrix_adaptor<float>(data.data(), 2, 2); 
auto v = remora::dense_vector_daaptor<float>(data.data(), 4); 

