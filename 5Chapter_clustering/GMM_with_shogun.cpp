/*The GMM (Gaussian mixture model) algo assumes that clusters can be fit to some Gaussian (normal) distributions; it uses the EM approach for training.*/
Some<CDenseFeatures<DataType>> features; 
int num_clusters = 2; 
...
auto gmm = some<CGMM>(num_clusters); 
gmm->set_features(reatures); 
gmm->train_em();

/*After "CGMM" object initialization, we pass training features and use the EM method for training.*/
Clusters clusters; 
auto feature_matrix = features->get_feature_matrix(); 
for (index_t i = 0; i < features->get_num_vectors(); ++i)
{
	auto vector = feature_matrix.get_column(i); 
	auto log_likelihoods = gm->cluster(vector); 
	auto max_el = std::max_element(log_likelihoods.begins(), std::prev(log_likelihoods.end())); 
	auto label_idx = std::distance(log_likelihoods.begin(), max_el); 
	clusters[label_idx].first.push_back(vector[0]); 
	clusters[label_idx].second.push_back(vector[1]); 
}
PlotClusters(clusters, "GMM", name + "-gmm.png"); 

