/*"Dlib::vector_normalizer_pca" type which objects can be used to perform PCA on user data. 
 * This implementation also normalizes the data. 
 * After we've instanctiated an object of this type, we use the "train()" method to fit the model to our data. 
 * The "train()" method takes "std::vectro" as samples and the "eps" value as paramters. 
 * The "eps" values controls how many dimensions should be preserved after the PCA has been transformed. */
void PCAReduction(const std::vector<Matrix> &data, double target_dim)
{
	Dlib::vector_normalizer_pca<Matrix> pca; 
	pca.train(data, target_dim / data[0],nr()); 

	std::vector<Matrix> new_data; 
	new_data.reserve(data.size()); 
	for (size_t i = 0; i < data.size(); ++i)
	{
		new_data.emplace_back(pca(data[i])); /*look out its called remember std::vector class and containors if you don't look at the C++ learning REPO*/
	}

	for (size_t r = 0; r < new_data.size(); ++r)
	{
		Matrix vec = new_data[r]; 
		double x = vec(0, 0); 
		double y = vec(1, 0); 
	}
}

/*After the algorithm has been trained, we use the object to transform individual samples. 
 * */
