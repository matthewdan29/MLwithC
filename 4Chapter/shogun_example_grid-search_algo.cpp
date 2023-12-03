/*The "CKernelRidgeRegression" class implements the poloynomial regression model based on the svm algo, and its uses kernel for precise model specialization.
 * That means we can use the polynomial kernel to simulate the polynomial regression model.*/

/*This is how you create a kernel type with configurable hyperparameters called polynomial degree.*/
auto kernel = some<CPolyKernel>(/*cache size*/ 256, /*degree*/ 15); 
kernel->init(x, x); 

/*The kernel object requires additional configuration, we add the normalization object to the kernel object*/
auto kernel_normaiizer = some<CSqrtDiagKernelNormalizer>(); 
kernel->set_normalizer(kernel_normiizer); 

/*"CKernelRidgeRegression" come with L2 regularization. 
 * below configure the initial values for regularization coefficent*/
float64_t tau_regularization = 0.00000001; 
float64_t tau_regularization_max = 0.000001; 
auto model = some<CKernelRidgeRegression>(tau_regularization, kernel, y); 

/*Next, we define the cross-validation object for the grid search. 
 * "CStratifiedCrossValidationSplitting" class implements the same size folds for splitting.*/
auto splitting_strategy = some<CStratifiedCrossValidationSplitting>(y, 5); 

/*MSE is used as performance metric "CMeanSquaredError" implements it*/
auto evaluation_criterium = some<CMeanSquaredError>(); 

auto cross_validation = some<CCrossValidation>(model, x, y, splitting_strategy, evaluation_criterium); 
cross_validation->set_autolock(false); 
cross_validation->set_num_runs(1);	/*Its configured one number of runs of the cross-validation process*/

/*"CModelSelectionParameters" object implements a node of a tree that contains a predefined range of values for one hyperparameter. 
 * Below method shows how to make a tree of such nodes, which will be analoy for the parameter grid*/
auto params_root = some<CModelSelectionParameters>();	/*I'm probly going to change the var "some" to "creampie_cumdump"*/
auto param_tau = some<CModelSelectionParameters>("tau");
params_root->append_child(param_tau); 
param_tau->build_values(tau_regularization, tau_regularization_max, ERangeType::R_Linear,tau_regularization_max); 
auto param_kernel = some<CModelSelectionParameters>("kernel", kernel); 
auto param_kernel_degree = some<CModelSelectionParameters>("degree");
param_kernel_degree->build_values(5, 15, ERangeType::R_LENEAR, 1); /*"build_values()" method generate a range of values*/
param_kernel->append_child(param_kernel_degree); 
params_root->append_child(param_kernel); 

/*Next, configure the cross-validation and the parameter grid objects, we can initialize and run the grid search algo. 
 * "CGridSearchModelSelection" class implements the grid search algo*/
auto model_selection = some<CGridSearchModelSelection>(cross_validation, params_root); 
auto best_parameters = model_selection->select_model(true); 	/*This method search for the best parameter values.*/
best_parameters->apply_to_machine(model); /*appling the best values to the model*/

/*last we need to retain the model*/
if (!model->train(x))
{
	std::cerr << "traing fail\n"; /*If you ever trained a ML model using the python tenserflow lib created by google then you know all though traing print out comes from a loop made in C++ just like this LOL! I find it funny*/
}
/*after a final training process the model is ready to be evaluated.*/
