/*This method was discribed in the paper "Chinese Whispers -- an Effecient Graph Clustering Algorithm and its Application to Natural Language Processing Problems by Chris Biemann"*/
/*"chinese_whispers()" function takes vector of weighted graph edges and outputs the container with cluster indices for eache of the vertices. 
 * For the performance consideration we limit the number of edges between dataset objects with a threeshold on distance. 
 * Moreover, as with the Newman Algorithm, this one also determines the number of resulting clusters by itself. */
std::vector<sample_pair> edges; 
for (long i = 0; i < inputs.nr(); ++i)
{
	for (long j = 0; j < inputs.nr(); ++j)
	{
		auto dist = length(subm(inputs, i, 0, 1, 2) - subm(inputs, j, 0, 1, 2)); 
		if (dist < 1)
			edges.push_back(sample_pair(i, j, dist)); 
	}
}
std::vector<unsigned long> clusters; 
const auto num_clusters = chinese_whispers(edges, clusters); 
