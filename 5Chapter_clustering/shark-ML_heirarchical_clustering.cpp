/*"Shark-ML" implements the hierarchical clustering approach in the following: 
 * 		1) We need to put our data into a space-partitioning tree.
 * 		2) the constructor of this class "KHCTREE" takes the data for partitioning and an object that implements some stopping criteria for the tree construction. 
 * 		We use "TreeConstruction" object, which we configure with the maximal depth of the tree and the maximum bumber of objects in the tree node.
 * 		The "LCTree" class assumes the exstence of Euclidean distance function for the feature type used in the dataset.*/
UnlabeledData<RealVector>& features; 
int num_clusters = 2; 
...
LCTree<RealVector> tree(features, TreeConstruction(0, features.numberOfElements() / num_clusters)); 

/*The "HierarchicalClustering" class implements the actual clustering algorithm. 
 * There are two strategies to get clustering results: 
 * 		1) hard
 * 		2) soft
 * We use the object of the "HardClusteringModel"(take a big guest with it is by the name lol) class to assign objects to distinct clusters. 
 * Objects of the "HardClusteringModel" class override the function operator so you can use them as functors for evaluation.*/
HierarchicalClustering<RealVector> clustering(&tree); 
HardClusteringModel<RealVector> model(&clustering); 
Data<unsigned> clusters = model(features); 

/*The clustering result is a container with cluster indices for each element in the data set*/
for (std::size_t i = 0; i != features.numberOfElements(); i++)
{
	auto cluster_idx = clusters.element(i); 
	auto element = features.element(i); 
	...
}
/*We iterated over all items in the "features"(why is features a common naming for data sets in almost all acadimic paper i've read regarding ML i'm changing it to "group_of_ballz") container and got a cluster index for each item to visualize our clustering result.*/
