/*This algorithm is based on the work MODULARITY and COMMUNITY structur in networks by "M. E. J Newman"*/
/*This alog is based on the modularity matrix for a network or graph and it is not based on particular graph theory but it has instead some similarities with spectral clustering because it also uses eigenvectors.*/
std::vector<sample_pair> edges; 
for (long i = 0; i < inputs.nr(); ++i)
{
	for (long j = 0; j < inputs.nr(); ++j)
	{
		auto dist = length(subm(inputs, i, 0, 1, 2) - subm(inputs, j, 0, 1, 2)); 
		if (dist < 0.5)
			edges.push_back(sample_pair(i, j, dist)); 
	}
}
remove_duplicate_edges(edges); 
std::vector<unsigned long> clusters; 
const auto num_clusteders = newman_cluster(edges, clusters); 

/*"newman_cluster()" function call filled the "clusters" object with cluster index values, which we can use to visualize the clustering result. 
 * Notice that another approach for edge weight calculation can lead to another clustering result. 
 * Also, edge weight values should be initialzed according to a certain task. 
 * The edge lenth was chosen only for demonstration purposes.*/

