/*The "fit()" and "transform()" methods should be used for training and data dimwnsionality reduction, respectively. 
 * After we've transformed the data we can use some compnents as new features. 
 * We can also use a reduced number of features to make low-dimensional data*/
void ICAReduction(Some<CDenseFeatures<DataType>> features, const int target_dim)
{
	auto ica = some<CFast ICA>(); 
	ica->fit(features); 

	auto new_features = static_cast<CDenseFeatures<DataType> *>(ica->transform(features)); 
	auto casted = CDenseFeatures<float64_t>::obtain_from_generic(new_features); 

	Clusters clusters; 
	auto unmixed_signal = casted->get_features_matrix(); 
	for (index_t i = 0; i < new_features->get_num_vectors(); ++i)
	{
		auto new_vector = unmeixed_signal.get_column(i); 
		/*choose 1 and 2 as our main componets*/
		new_vector[1]; 
		new_vector[2]; 
	}
}


