/*First, we use "createCVSameSize" function which define the partitions of our dataset, which are five chunks of the same size.*/
cont unsigned int num_folds = 5;	/*this explains itself*/
CVFolds<RegressionDataset> folds = createCVSameSize<RealVector, RealVector>(train_data, num_folds); 

/*Next, we initialize and configure our model. 
 * The model parameters are usually configured with trainer objects, which pass them to the model object*/
double regularization_factor = 0.0; 
double polynomial_degree = 8; 
int num_epochs = 300; 
PolynomialModel<> model; 
PolynomialRegression trainer(regularization_factor, polynomial_degree, num_epochs); 

/*Now that we have the trainer, model, and folds objects, we initialize the "CrossValidationError" object. 
 * As a performance meric, "AbsoluteLoss" object which implements the MAE metric*/
AbsoluteLoss<> loss; 
CrossValidationError<PolynomialModel<>, RealVector> cv_error(folds, &trainer, &model, &trainer, &loss); 

/*"GridSearch" performs the grid search algo. 
 * We should configure the object of this class with the parameter ranges. 
 * There is the "configure()" method, which takes three containers as arguments. 		1) Specifies the minimum values for each parameter range, 
 * 		2) Specifies the maximum values for each parameter rane 
 * 		3) Specifies the number of values in each parameter range. 
 * Notice that the order of the parameters in the range containers should be the same as how they were defined in the trainer class.*/
GridSearch grid; 
std::vector<double> min(2); 
std::vector<double> max(2); 
std::vector<size_t> sections(2); 
/*regularization factor*/
min[0] = 0.0; 
max[0] = 0.00001; 
sections[0] = 6; 
/*polynomial degree*/
min[1] = 4; 
max[1] = 10.0; 
sections[1] = 6; 
grid.configure(min, max, sections); 

/*After initializing the grid, we can use the "step()" method to perform the grid search for the best hyperparameter values; this method should be called only once. 
 * Now we should retain our model with the parameters we found*/
grid.step(cv_error); 

trainer.setParameterVector(grid.solution().point); 
trainer.train(model, train_data); 


