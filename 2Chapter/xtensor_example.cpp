/*"xtensor" library is inspired by NumPy, (its the tenserflow lib from python but in C++) 
 * ML algorithms are mainly described using Python and NumPy, so this library can make it easier to move them to C++.*/

/*The "xarray" type is a dynamically sized multidimensional array, below shows the methods*/
std::vector<size_t> shape = {3, 2, 4}; 
xt::xarray<double, xt::layout_type::row_major> a(shape); 

/*"xtensor" type is a multidimensional array whose dimensions are fixed at compilation time. 
 * Exact dimension values can be configured in the initialization step, look below*/
std::array<size_t, 3> shape = {3, 2, 4}; 
xt::xtensor<double, 3> a(shape); 

/*"xtensor_fixed" type is a multidimensional array with a dimension shape fixed at compi le time, */
xt::xtensor_fixed<double, xt::xshape<3,2,4>> a; 

/*Initialization of "xtensor" arrays can be done with C++ initializer lists, like below*/
xt::xarray<double> arr1{{1.0, 2.0, 3.0}, 
			{2.0, 5.0, 7.0}, 
			{2.0, 5.0, 7.0}}; /*initialize a 3x3 array*/

/*"xtensor" library also has builder function for special tensor types.*/
std::vector<uint64_t> shape = {2, 2}; 
xt::ones(shape); 
xt::zero(shape); 
xt::eye(shape); 	/*matrix with ones on the diagonal */

/*We can map existing C++ arrays into the "xtensor" container with the "xt::adapt" function.
 * This function returns the object that uses the memory and values from the underlying object*/
std::vector<float> data{1,2,3,4}; 
std::vector<size_t> shape{2,2}; 
auto data_x = xt::adapt(data, shape); 

/*You can use direct access to container elements, with the "()" operator, to set or change tensor values*/
std::vector<size_t> shape = {3,2,4}; 
xt::xarray<float> a = xt::ones<float>(shape); 
a(2,1,3) = 3.14f; 

/*Below shows the use of arithmetic operations with the "xtensor" library*/
auto a = xt::random::rand<double>({2,2}); 
auto b = xt::random::rand<double>({2,2}); 
auto c = a + b; 
a -= b; 
c = xt::linalg::dot(a,b); 
c = a + 5; 

/*To get partial access to the "xtensor" containers, we can use the "xt::view" function.*/
xt::xarry<int> a{{1, 2, 3, 4}, 
		{5, 6, 7, 8}, 
		{9, 10, 11, 12}, 
		{13, 14, 15, 16}}; 
auto b = xt::view(a, xt::range(1,3), xt::range(1,3)); 

/*"xtensor" library implments automatic broadcasting in most cases. 
 * When the operation involves two arrays of differnt dimensions, it transmit the array with the smaller dimension across the leading dimension of the other array, so we can directly add vector to a matrix.*/
auto m = xt::random::rand<double>({2,2}); 
auto v = xt::random::rand<double>({2,1}); 
auto c = m + v; 

