/*Shark-ML lib uses "kMeans()" to implement the k-means algo which take 3 parameters: 
 * 		1) The trainig dataset
 * 		2) the desired number of clusters
 * 		3) the output parameterfor cluster centriods*/
UnlabeledData<RealVector> features; 
int num_clusters = 2; 
...
Centroids centroids; 
kMeans(features, num_clusters, centroids); 

/*After we get the centroids, we can initialize an object of the "HardClusteringModel" class.*/
HardClusteringModel<RealVector> model(&centroids); 
Data<unsigned> clusters = model(features); 

for (std::size_t i = 0; i != features.numberOfElements(); i++)
{
	auto cluster_idx = clusters.element(i); 
	auto element = features.element(i); 
	...
}

/*After we used the "model" object as a functor to preform clustering. 
 * The result was a container with cluster indices for each element of the input dataset.
 * We use these cluster indices to visualize the final reuslt.*/
