/*"Dlib" provides the "krr_trainer" class wich can get the template argument of the "linear_kernel" type to solve linear regression tasks. 
 * This class implements direct analytical solving for this type of problem with kernel ridge regression algorithm*/
std::vector<matrix<double>> x; 
std::vector<float> y; 
krr_trainer<KernelType> trainer; 
trainer.set_kernel(KernelType()); 
decision_function<KernelType> df = trainer.train(x, y); 

/*For new x inputs, we can predict new y values */
std::vector<matrix<double>> new_x; 
for (auto& v : x)
{
	auto prediction = df(v); 
	std::cout << prediction << std::endl; 
}
