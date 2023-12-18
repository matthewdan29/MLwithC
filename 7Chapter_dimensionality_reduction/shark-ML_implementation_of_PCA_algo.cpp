/*We us "encoder()" method for objects of this class should be configured with the number of target dimensions. 
 * This method takes 2 parameters. 
 * 		1) first one reference to the object of the "LinearModel" class
 * 		2) THe second one is the number of target dimensions. 
 * After the object of the "LinearModel" class has been configured, it can be used for data transformation regarding the functional object. 
 * Its called result is a new object of the "Data<>RealVector" class*/
void PCAReduction(const UnlabeledData<RealVector> &data, const UnlabeledData<RealVector>& lables, size_t target_dim)
{
	PCA pca(data); 
	LinearModel<> encoder; 
	pca.encoder(encoder, target_dim); 
	auto new_data = encoder(data); 

	for (size_t i = 0; i < new_data.numberOfElements(); ++i)
	{
		auto x = new_data.element(i)[0]; 
		auto y = new_data.element(i)[1]; 
	}
}
