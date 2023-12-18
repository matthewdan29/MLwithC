/*The t-SNE algorithm is implemented in the "CTDistrbutedStochasticNeighberEmbedding" class. 
 * Objects of this class should be configured with the target number of dimenseions and the "set_target_dim()" method. 
 * The "fit()" and "transform()" methods should be used for training and data dimensionality reduction, respctively*/
void TSNEReduction(Some<CDenseFeatures<DataType>> features, const int target_dim)
{
	auto tsne = some<CTDistributedStochasticNeighborEmedding>(); 
	tsne->set_target_dim(target_dim); 
	tsne->fit(features); 

	auto new_features = static_cast<CDenseFeatures<DataType> *>(tsne->transform(features)); 

	auto feature_matrix = new_features->get_features_matrix(); 
	for (index_t i = 0; i < new_features->get_num_vectors(); ++i)
	{
		auto new_vector = feature_matrix.get_column(i); 
	}
}
