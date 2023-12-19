/*Here we are using "CGaussianKernel" class who's parameters for configuration, but we used only one named the "combined_kernel_weight" paramter because it gave the most reasonable configuration for our model after a series of experiments(not my words)*/
Some<CDenseFeatures<DataType>> features; 
Some<CMulticlassLabels> labels; 
Some<CDenseFeatures<DataType>> test_features; 
Some<CMulticlassLabels> test_labels; 

/*These are our train and test dataset objects' definition*/
auto kernel = some<CGaussianKernel>(features, features, 5); 
auto svm = some<CMulticlassLibSVM>();		/*one vs one classification*/
svm->set_kernel(kernel); 

/*Using these datasets, we initialized the "CMulticlassLibSVM" class object and configured its kernel*/
auto root = some<CModelSelectionParameters>(); 	/*Search for hyper-parameters*/
CModelSelectionParamters* c = new CModelSelectionParameters("C");	/*C - how much you want to avoid missclassifying*/
root->append_child(c); 
c->build_values(1.0, 1000.0, R_LINEAR, 100.); 

auto params_kernel = some<CModelSelectionParameters>("kernel", kernel); 
root->append_child(params_kernel); 

auto params_kernel_width = some<CModelSelectionParameters>("combined_kernel_weight"); 
params_kernel_width->build_values(0.1, 10.0, R_LINEAR, 0.5); 

params_kernel->append_child(params_kernel_width); 

/*Then, we configured cross-validation parameters object to look for the best hyperparameters combination*/
index_t k = 3; 
CStratifiedCrossValidationSplitting* splitting = new CStratifiedCrossValidationSplitting(labels, k); 

auto eval_criterium = some<CMutliclassAccuracy>(); 

auto cross = some<CCrossValidation>(svm, features, labels, splitting, eval_criterium); 

cross->set_num_runs(1); 

auto model_selection = some<CGridSearchModelSelection>(cross, root); 
CParameterCombination* best_params = wrap(model_selection->select_model(false)); 
best_params->apply_to_machine(svm); 
best_params->print_tree(); 

/*Having configured the cross-validation parameters, we initialized the "CCrossValidation" class object and ran the grid-search process for model selection*/

/*train SVM*/
svm->set_labels(labels); 
svm->train(features); 

/*evaluate model on test data*/
auto new_labels = wrap(svm->apply_multiclass(test_features)); 

/*estimate accuracy*/
auto accuracy = eval_criterium->evaluate(new_labels, test_labels); 
std::cout << "svm " << name << " accuracy = " << accuracy << std::endl; 

/*process results*/
auto feature_matrix = test_features->get_feature_matrix(); 
for (index_t i = 0; i < new_labels->get_num_labels(); ++i)
{
	auto label_idx_pred = new_labels->get_num_label(i); 
	auto vector = feature_matrix.get_column(i); 
	...
}

/*When the best hyperparameters were found and applied to the model we repeated training and did the evalution. */

