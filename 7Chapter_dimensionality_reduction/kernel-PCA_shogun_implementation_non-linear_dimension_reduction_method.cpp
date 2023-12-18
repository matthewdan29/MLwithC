/*The non-linear version of PCA in the shogun library is implemented in the "CKernelPCA" class. 
 * The main difference is that its configured with an additional method "set_kernel()", which should be used to pass the pointer to the specific kernel object.*/
void KernelPCAReduction(Some<CDenseFeatures<DataType>> features, const int target_dim)
{
	auto gauss_kernel = some<CGaussianKernel>(features, features, 0.5); 
	auto pca = some<CKernelPCA>(); 
	pca->set_kernel(gauss_kernel.get()); 
	pca->set_target_dim(target_dim); 
	pca->fit(features); 

	auto feature_matrix = features->get_features_matrix(); 
	for (index_t i = 0; i < features->get_num_vectors(); ++i)
	{
		auto vector = feature_matrix.get_column(i); 
		auto new_vector = pca->apply_to_feature_vector(vector); 
	}
}
