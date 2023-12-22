/*"CRandomForest" class implements the random forest algo with the 2 parameters: 
 * 		1) The number of trees
 * 		2) The number of attributes chosen randomly during the node splitting when the algorithm builds a tree*/
/*"set_combination_rule" is the rule on how the tree results shoul dbe combined into one final answer. 
 * "CMajorityVote" class implements the majority vote scheme. */
/*"set_machine_problem_type" method of the "CRandomForest" class is configured to what type of problem we want to solve with the random forest. */
/*For training we will use the "set_labels" and the "train" methods with appropriate parameters, as well as an object of the "CRegressionLabels" type, and an object of the "CDenseFeatures" type. 
 * "apply_regreesion" method for the evaluation.*/
void RFClassification(Some<CDenseFeatures<DataType>> features, Some<CRegressionLabels> labels, Some<CDenseFeatures<DataType>> test_features, Some<CRegressionLabels> test_labels)
{
	int32_t num_rand_feats = 1; 
	int32_t num_bags = 10; 

	auto rand_forest = shogun::some<CRandomForest>(num_rand_feats, num_bags); 
	auto vote = shogun::some<CMajorityVote>(); 
	rand_forest->set_combination_rule(vote); 
	/*mark feature type as continuous*/
	SGVector<bool> feature_type(1); 
	feature_type.set_const(false); 
	rand_forest->set_reature_types(feature_type); 

	rand_forest->set_labels(labels); 
	rand_forest->set_machine_problem_type(PT_REGRESSION); 
	rand_forest->train(features); 

	/*evaluate model on test data*/
	auto new_labels = wrap(rand_forest->apply_regression(test_features)); 

	auto eval_criterium = some<CMeanSquaredError>(); 
	auto accuracy = eval_criterium->evalueate(new_labels, test_labels); 
	...
}
