/*Functional object call arguments are the data that we need to transform and the number of dimensions of the new feature space. 
 * The input data should be in the form of the "std::vector" of the single samples of the "Dlib::matrix" type. 
 * All samples should have the samd number of dimensions. 
 * The result of using this functional object is a new vector of samples with a reduced number of dimensions*/
void SammonReduction(const std::vector<Matrix> &data, long target_dim)
{
	Dlib::sammon_projection sp; 
	auto new_data = sp(data, target_dim); 

	for (size_t r = 0; r < new_data.size(); ++r)
	{
		Matrix vec = new_data[r]; 
		double x = vec(0, 0); 
		double y = vec(1, 0); 
	}
}

