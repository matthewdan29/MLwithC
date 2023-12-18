/*The PCA algorithm is implemented in the "CPCA" class. 
 * It has one primary configuration option -- the number of target dimensions, which can be modified with the "set_target_dim()" method. 
 * After we make this configuration we need to execute the "fit()" method for training purposes and then use the "apply_to_feature_vector()" method to transform an individual sample*/

void PCAReduction(Some<CDenseFeatures<DataType>> features, const int target_dim)
{
	auto pca = some<CPCA>(); 
	pca->set_target_dim(target_dim); 
	pca->fit(features); 

	auto feature_matrix = features->get_feature_matrix();
	for (index_t i = 0; i < features->get_num_vectors(); ++i)
	{
		auto vector = feature_matrix.get_column(i); 
		auto new_vector = pca->apply_to_feature_vector(vector); 
	}
}
