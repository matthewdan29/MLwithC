/*The "Dlib" library has the "Dlib::mat()" function for wrapping C++ containers into the "Dlib" matrix object. 
 * It also take a pointer to the data and matrix dimensions as arguments.*/
double data[] = {1,2,3,4,5,6}; 
auto m2 = Dlib::mat(data, 2, 3); 	/*Create matrix with size 2x3*/
