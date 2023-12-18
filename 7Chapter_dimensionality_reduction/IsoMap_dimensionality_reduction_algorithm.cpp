/*Objects of this class should be configured with the target number of dimensions and the number of neighbors for graph construction. 
 * The "set_target_dim()" and "set_k()" methods should be used for this. 
 * The "fit()" and "transform()" methods should be used for the training and data dimensionality reduction, respectively*/
void IsoMapReduction(Some<CDenseFeatures<DataType>> features, const int target_dim)
{
	auto IsoMap = some<CIsoMap>(); 
	IsoMap->set_target_dim(target_dim); 
	IsoMap->set_k(100); 
	IsoMap->fit(features); 

	auto new_features = static_cast<CDenseFeatures<DataType> *>(IsoMap->transform(features)); 

	auto feature_matrix = new_features->get_feature_matrix(); 
	for (index_t i = 0; i < new_features->get_num_vectors(); ++i)
	{
		auto new_vector = feature_matrix.get_column(i); 
	}
}
