/*Assume we have the following train and test data: */
Some<CDenseFeatures<DataType>> features; 
Some<CMulticlassLabels> labels; 
Some<CDenseFeatures<DataType>> test_features; 
Some<CMulticlassLabels> test_labels; 

/*As we decided to use a cross-validation process, lets define the required objects*/
auto root = some<CModelSelectionParameters>(); 	/*search for hyper-parameters*/
CModelsSelectionParameters* z = new CModelSelectionParameters("m_z"); 	/*z - regularization parameter*/
root->append_child(z); 
z->build_values(0.2, 1.0, R_LINEAR, 0.1); 

/*Every trainable model in the "Shogun" library has the "print_model_params()" method, which prints all model parameters available for automatic configuration with the "CGridSearchModelSelection" class, so its useful to check exact parameter names. */
index_t k = 3; 
CStratifiedCrossValidationSplitting* splitting = new CStratifiedCrossValidationSplitting(labels, k); 

auto eval_criterium = some<CMulticlassAccuracy>(); 

auto log_reg = some<CMulticlassLogisticRegression>(); 
auto cross = some<CCrossValidation>(log_reg, features, labels, splitting, eval_criterium); 

cross->set_num_runs(1); 

/*we configured the instance of the "CCrossValidation" class, which instances of a splitting strategy and an evalution criterium object, as well as training features and labels for initialization. 
 * The splitting strategy is defined by the instance of the "CStratifedCrossValidationSplitting" class and evaluation metric. 
 * We used the instance of the "CMulticlassAccuracy" class as an evalution criterium*/
auto model_selection = some<CGridSearchModelSelection>(cross, root); 
CParameterCombination* best_params = wrap(model_selection->select_model(false));
best_params->apply_to_machine(log_reg); 
best_params->print_tree(); 

/*After we configured the cross-validation objects, we used it alongside the parameters tree to initialize the instance of the "CGridSearchModelSelection" class and then we used it method to search for the best model parameters*/
/*This method returned the instance of the "CParameterCombination" class which used the "apply_to_machine()" method for the initialization of model parameters with this objects's specific values*/

/*Train*/
log_reg->set_labels(labels); 
log_reg->train(features); 

/*evaluate model on test data*/
auto new_labels = wrap(log_reg->apply_multiclass(test_features)); 

/*estimate accuracy*/
auto accuracy = eval_criterium->evaluate(new_labels, test_labels); 

/*process results*/
auto feature_matrix = test_features->get_feature_matrix(); 
for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
{
	auto label_idx_pred = new_labels->get_label(i); 
	auto vector = feature_matrix.get_column(i); 
	...
}
/*After we found out the best parameters, we trained our model on the full training dataset and evaluated it on the test set. 
 * The "CMulticlassLogisticRegression" class has a method named "apply_multiclass()" that we used for a model evaluation on the test data. 
 * This method returned an object of the "CMulticlassLabels" class. 
 * The "get_label()" method was then used to access labels values. */
