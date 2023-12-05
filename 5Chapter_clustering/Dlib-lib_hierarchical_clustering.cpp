/*"bottom_up_cluster()" function takes the matrix of distances between dataset objects, the cluster indices container, and the number of clusters as input parameters. 
 * Notice that it returns the container with cluster indices in the order of distances provided in the matrix 
 * Next we will fill the distance matix with pairwise Euclidean distances between each pair of elements in the input data set*/
matix<double> dists(inputs.nr(), inputs.nr()); 
for (long r = 0; r < dists.nr(); ++r)
{
	for (long c = 0; c < dists.nc(); ++c)
	{
		dists(r, c) = length(subm(inputs, r, 0, 1, 2) - subm(inputs, c, 0, 1, 2)); 
	}
}
std::vector<unsigned long> clusters; 
bottom_up_cluster(dists, clusters, num_clusters); 
