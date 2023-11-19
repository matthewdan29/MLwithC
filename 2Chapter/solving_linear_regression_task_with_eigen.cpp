/*"LeastSquaresConjugateGradient" class has two main settings:
 * 	1) The maximum number of iterations and a tolerance threshold value that is used as a stopping criteria as an upper bound to the relative residual error*/
typedef float DType; 
using Matrix = Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic>; 
int n = 10000; 
Matrix x(n,1); 
Matrix y(n,1); 
Eigen::LeastSquaresConjugateGradient<Matrix> gd; 
gd.setMaxIterations(1000); 
gd.setTolerance(0.001); 
gd.compute(x); 
auto b = dg.solve(y); 

/*For new x inputs, we can predict new 'y' values with matrices operations,*/
Eigen::Matrixxf new_x(5,2); 
new_x << 1, 1, 1, 2, 1, 3, 1, 4, 1, 5; 
auto new_y = new_x.array().rowwise() * b.transpose().array(); 

/*We can calculate parameter's b vector by solving the normal equation directly*/
auto b = (x.transpose() * x).ldlt().solve(x.transpose() * y); 


