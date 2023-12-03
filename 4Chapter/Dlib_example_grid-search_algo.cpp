/*"Dlib" has asll the necessary functionality for the grid search algo
 * "CrossValidationScore" function is used for cross-validation and returns the value of the performance metric*/
auto CrossValidationScore = [&](const double gamma, const double c, const double degree_in)
{
	auto degree = std::floor(degree_in); 
	using KernelType = Dlib::polynomail_kernel<SampleType>; 
	Dlib::svr_trainer<KernelType> trainer; 		/*"svr_trainer" class implements kernel ridge regression based on the SVM algo*/
	trainer.set_kernel(KernelType(gamma, c, degree)); 
	Dlib::matrix<double> result = Dlib::cross_validate_regression_trainer(trainer, samples, raw_labels, 10);	/*"cross_validate_regression_trainer()" function returns the matrix, along with the values of different performance metrics*/
	return result(0, 0); 
}; 

/*Next, we can search for the best parameters that were set with the "find_min_global" function*/
auto result = find_min_global(CrossValidationScore, {0.01, le-8, 5}, {0.1, 1, 15}, max_function_calls(50));		/*The first 2 parameters are (minimum values for gamma, c, and degree) (maximum values for gamma, c, and dgree)*/

/*Next, we can extract the best hyperparameters and train our model with them*/
double gamma = result.x(0); 
double c = result.x(1); 
double degree = result.x(2); 
using KernelType = Dlib::polynomial_kernel<SampleType>; 
Dlib::svr_trainer<KernelType> trainer; 
trainer.set_kernel(KernelType(gamma, c, degree)); 
auto descision_func = trainer.train(samples, raw_labels); 

