/*The most widely used kernel is based on the Gaussian distribution and is known as the Radical Basis Kernel. 
 * It is implemented in the "radial_basis_kernel". 
 * We represent datasets in the "Dlib" library as "vector" of separated samples. 
 * Befor using this trainer object we have to convert a matrix dataset into a vector*/
void OneClassSvm(const Matrix& normal, const Matrix& test)
{
	typedef matrix<double, 0, 1> sample_type; 
	typedef radial_basis_kernal<sample_type>kernel_type; 
	svm_one_class_trainer<kernel_type> trainer; 
	trainer.set_nu(0.5); 		/*control smoothness of the solution*/
	trainer.set_kernel(kernel_type(0.5));		/*kernel bandwidth*/
	std::vector<sample_type> samples; 
	for (long r = 0; r < normal.nr(); ++r)
	{
		auto row = rowm(normal, r); 
		sample.push_back(row); 
	}
	decision_function<kernel_type> df = trainer.train(samples); 
	Clusters cluster; 
	double dist_threshold = -2.0; 

	auto detect = [&](auto samples)
	{
		auto row = dlib::rowm(samples, r); 
		auto dist = df(row); 
		if (p > dist_threshold)
		{
			/*do something with anomalies */
		} else 
		{
			/*Do something with norma*/
		}
	}; 
	
	detect(normal); 
	detect(test); 
}

/*The result of the training process is a decision function object of the "decision_function<kernel_type>" class that we can use for single sample classification. 
 * Objects of this type can be used as a regular function. 
 * The result of a decision function is the distance from the normal class boundary, so the most distant samples can be classified as anomalies. */
