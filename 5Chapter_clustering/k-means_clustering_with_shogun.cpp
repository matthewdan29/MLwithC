/*"CKMeans" class constructor takes two parameters: 
 * 		1) the number of cluster 
 * 		2) the object for distance measure calculation.
 * Below we use the distance object defined with the "CEuclibeanDistance" class. 
 * After we construct the object of the "CKMeans" type, we use the "CKMeans::train()" method to train our model on our training set*/
Some<CDenseFeatures<DataType>> features; 
int num_clusters = 2; 
...
CEuclideanDistance* distance = new CEuclideanDistance(features, features); 
CKMeans* clustering = new CKMeans(num_clusters, distance); 
clustering->train(features); 

/*When we have trained the k-means object, we can use the "CKMeans::apply()" method to classify the input dataset. 
 * if you use this method with arguments, the training dataset is used for classification. 
 * The result of applying classification is a container object with labels. 
 * We can cast it to the "CMulticlassLabels" type for more natural use. 
 * Below shows how to classify the input data and also plots the results of clustering.*/
Clusters clusters; 
auto feature_matrix = features->get_feature_matrix(); 
CMulticlassLabels* result = clustering->apply()->as<CMulticlassLabel>(); 
for (index_t i = 0; i < result->get_num_labels(); ++i)
{
	auto label_idx = result->get_label(i); 
	auto vector = feature_matrix.get_column(i); 
	clusters[label_idx].first.push_back(vector[0]); 
	clusters[label_idx].second.push_back(vector[1]); 
}
PlotClusters(clusters, "K-Means", name + "-kmeans.png"); 


