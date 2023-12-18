/*"set_target_dim()" method should be used to modify the value of the target dimensions, while the "fit()" and "transform()" methods should be usded for training and data dimensionality reduction, respectively*/
void FAReduction(Some<CDenseFeatures<DataType>> features, const int target_dim)
{
	auto fa = some<CFactorAnalysis>(); 
	fa->set_target_dim(target_dim); 
	fa->fit(features); 

	auto new_features = static_cast<CDenseFeatures<DataType> *>(fa->transform(features)); 

	auto feature_matrix = new_features->get_feature_matrix(); 
	for (index_t i = 0; i < new_features->get_num_vectors(); ++i)
	{
		auto new_vector = feature_matrix.get_column(i); 
	}
}

