/*Before using this algorithm, we have to calculate the distance between all features in the training dataset. 
 * This operation can be done, with the instance of the "CEuclideanDistance" class, which implements the "CDistance" interface. 
 * After we have the object containing the distances for our training set we can initialize the object of the "CKNN" class, which takes the distance object, training labels, and the k parameter. 
 * This object uses the "train()" method to perform model training.*/
void KNNClassification(Some<CDenseFeatures<DataType>> features, Some<CMulticlassLabels> labels, Some<CDenseFeatures<DataType>> test_features, Some<CMulticlassLabels> test_labels)
{
	int32_t k = 3; 
	auto distance = some<CEuclideanDistance>(features, features); 
	auto knn = some<CKNN>(k, distance, labels); 
	knn->train(); 

	/*evaluate model on test data*/
	auto new_labels = wrap(knn->apply_multiclass(test_features)); 

	/*estimate accuracy*/
	auto eval_criterium = some<CMulticlassAccuracy>(); 
	auto accuracy = eval_criterium->evaluate(new_labels, test_labels); 

	/*process results*/
	auto feature_matrix = test_features->get_feature_matrix(); 
	for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
	{
		auto label_idx_pred = new_labels->get_label(i); 
		...
	}
}

/*after the model is trained, we can use the already known "apply_multiclass()" method for evaluation.*/
