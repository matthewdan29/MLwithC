/*The gradient boosting algorithm is implemented in the "CStochasticGBMachine" class. 
 * The main parameters to configure "CStochasticGBMachine" are: 
 * 		1) The base ensemble algorithm model and 
 * 		2) the loss function
 * 		3) number of iterations, 
 * 		4) the learning rate, 
 * 		5) the fraction of training vectors to be chosen randomly at each iteration. */
/*The implementation of the decision tree algorithm in the lib can be found in the "CCARTree" class. 
 * A classification and regression tree (CART) is a binary decision tree that is constructed by splitting a node into two child nodes repeatdly, beinginning with the rood node that contains the whole dataset. */
/*For configuation of "CCARTree" type object the constructor of this object takes: 
 * 		1) vector of the feature type 
 * 		2) the problem type 
 * 		3) (after the object is constructed) configure the tree depth*/
/*Then, we have to create the loss function object which will be "CSquaredLoss"*/
/*For training we have to use the "set_labels" and "train" methods: the object "CRegressionLabels" type and the object of the "CDenseFeatures" type respectively. 
 * For evaluation, the "apply_regression" method can be used*/

void GBMClassification(Some<CDenseFeatures<DataType>> features, Some<CRegressionLabels> labels, Some<CDenseFeatures<DataType>> test_features, Some<CRegressionLabels> test_labels)
{
	/*mark feature type as continous*/
	SGVector<bool> feature_type(1); 
	feature_type.set_const(false); 

	auto tree = some<CCARTree>(feature_type, PT_REGRESSION); 
	tree->set_max_depth(3); 
	auto loss = some<CSquaredLoss>(); 

	auto sgbm = some<SCtochasticGBMachine>(tree, loss, /*iterations*/100, /*learning rate*/ 0.1, /*sub-set fraction*/ 1.0); 
	sgbm->set_labels(labels); 
	sgbm->train(features); 

	/*evaluate model on test data*/
	auto new_labels = wrap(sgbm->apply_regression(test_features)); 

	auto eval_criterium = some<CMeanSquaredError>(); 
	auto accuracy = eval_criterium->evaluate(new_labels, test_labels); 
	...
}
