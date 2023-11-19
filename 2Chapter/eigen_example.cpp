/*"Eign" is a general purpose linear algebra C++ library. */

/*We can define the type for a matrix with known dimensions and floating point data type like below*/
typedef Eigen::Matrix<float, 3, 3> MyMatrix33f; 


/*We can define a vector in the following way below*/
typedef Eigen::Matrix<float, 3, 1> MyVector3f; 


/*We can define matrix types that will take the number of rows or columns at initialization during runtime. 
 * To define such types, we can use a special type variable for the "Matrix" class template argument named "Eigen::Dynamic". 
 * To define a matrix of doubles with dynamic dimensions, method below*/
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MyMatrix; 

/*Objects initialized from the types we defined will look like below*/
MyMatrix33f a; 
MyVector3f v; 
MyMatrix m(10,15); 

/*To put some values into these objects, we can use several approcahes.
 * We can use special predefined initialization function, as below*/
a = MyMatrix33f::Zero();	/*Fill matrix elements with zeros*/
a = MyMatrix33f::Identity(); 	/*fill matrix as Identity matrix*/
v = MyVector3f::Random(); 	/*fill matrix elements with random values*/

/*We can use the comma-initializer syntax, like below*/
a << 1,2,3,
     4,5,6, 
     7,8,9; 

/*direct element access is as followed for matrix coefficients.*/
a(0,0) = 3; 

/*We can use the object of the "Map" type to wrap an existent C++ array or vector in the "Matrix" type object. 
 * This kind of mapping object will use memory and values from the underlying object, and will not allocate the additional memory and copy the values. 
 * Below is a snippet showing you the "Map" type*/
int data[] = {1, 2, 3, 4}; 
Eigen::Map<Eigen::RowVectorxi> v(data,4); 
std::vector<float> data = {1,2,3,4,5,6,7,8,9}; 
Eigen::Map<MyMatrix33f> a(data.data()); 

/*We can use initialized matrix objects in mathematical operations. 
 * Matrix and vector arithmetic operations in the "Eigen" library are offered either through overloads of standard C++ arithmetic operations. 
 * Below shows general math operations in "Eigen"*/

using namespace Eigen; 
auto a = Matrix2d::Random(); 
auto b = Matrix2d::Random(); 
auto result = a + b; 
result = a.array() * b.array(); 	/*element wise multiplication*/
result = a.array() / b.array(); 	
a += b; 
result = a * b; 			/*matrix multiplication*/
a = b.array() * 4; 			/*Also its possible to use scalars*/


/*For performing operations on only a part of the matix. 
 * "Eigen" provides the "block" method, which takes for parameters: "i, j, p, q". 
 * These parameters are the block size "p,q" and the starting point "i,j". 
 * Below is the method */
Eigen::Matrixxf m(4,4); 
Eigen::Matrix2f b = m.block(1,1,2,2); 	/*Copying the middle part of the matrix*/
m.block(1,1,2,2,) *= 4; 		/*Change values in original matrix*/


/*There are two more methods to access rows and columns by index, which are also a type of block operation. 
 * below shows the use of "col()" and "row()" methods*/
m.row(1).array() += 3; 
m.col(2).array() /= 4; 

/*"Eigen" support broadcasting with the "colwise()" and "rowwise()" methods. 
 * Broadcasting can be interpreted as a matrix by replication it in one direction. */
Eigen::Matrixxf mat(2,4); 
Eigen::Vectorxf v(2); 			/*colum vector*/
mat.colwise() += v; 
