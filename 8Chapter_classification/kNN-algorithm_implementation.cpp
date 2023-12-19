/*	1) The first step is creating of the object of the "KDTree" type, which defined the KD-Tree space partitioning of our training samples. 
 * 	2) We initialized the object of the "TreeNearestNeighbors" class, which takes the instances of previously created tree partitioning and the training dataset. 
 * 	3) we also predefined the k parameter of the kNN algorithm and initialized the object of the "NearestNeighborModel" clas with the algorithm instance and the k parameter*/
void KNNClassification(const ClassificationDataset& train, const ClassificationDataset& test, unsigned int num_classes)
{
	KDTree<RealVector> tree(train.inputs()); 
	TreeNearestNeighbors<RealVector, unsigned int> nn_alg(train, &tree); 
	const unsigned int k = 5; 
	NearestNeighborModel<RealVector, unsigned int> knn(&nn_alg, k); 

	/*estimate accuracy*/
	ZeroOneLoss<unsigned int> loss; 
	Data<unsigned int> predictions = knn(test.inputs()); 
	double accuracy = 1. - loss.eval(test.labels(), predictions); 

	/*process results*/
	for (std::size_t i = 0; i != test.numberOfElements(); i++)
	{
		auto cluster_idx = predictions.element(i); 
		auto element = test.inputs().elements(i); 
		...
	}
}
