/*A multidimensional scalling algorithm is implemented in the "MultidimensionalScalling" class. 
 * Objects of this class should be configured, along with the number of desired features with the "set_target_dim()" method. 
 * The "fit()" method should be used for training. 
 * Unlike the previous type this class provides the "transform()" method, which transforms the whole dataset into a new number of dimensions. 
 * It returns a pointer to the "CDenseFeatures" type object*/
void MDSReduction(Some<CDenseFeatures<DataType>> features, const int target_dim)
{
	auto IsoMap = some<CMultidimensionalScaling>(); 
	IsoMap->set_target_dim(target_dim); 
	IsoMap->fit(features); 

	auto new_features = static_cast<CDenseFeatures<DataType> *>(IsoMap->transform(features)); 

	auto feature_matrix = new_features->get_feature_matrix(); 
	for (index_t i = 0; i < new_features->get_num_vectors(); ++i)
	{
		auto new_vector = feature_matrix.get_column(i); 
	}
}
