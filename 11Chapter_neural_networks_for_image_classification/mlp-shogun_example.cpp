/*To add a new layer example below*/

/*create the initial object*/
auto layers = some<CNeuralLayers>(); 
/*add the input layer*/
layers = wrap(layers->input(dimensions)); 
/*add the hidden layer*/
layers = wrap(layers->logistic(32)); 

/*Each time we add a new layer, we rewrite the pointer to the "CNeuralLayers" type object. 
 * We have to call the "done" method of the "CNeuralLayers" class after all the layers have been added. 
 * Then it returns an array of configured layers, which can be used to create the "CNeuralNetwork" type object. 
 * The "CNeuralNetwork" class implements functionality for network initialization and training. 
 * After we've created the "CNeuralNetwork" object, we have to connect all the layers by calling the "quick_connect" method. 
 * Then we can initialize the weights of all the layers by calling the "initialize_neural_network" method. 
 * This method can take an optional parameter, "sigma", which is the standard deviation of the Gaussian that's used to initialize the parameters randomly. */

/*we chose the gradient descent method by calling "set_opimzation"* method with the "NNOM_GRADIENT_DESCENT" enumeration value argument. 
 * Other settings are standard for the gradient descent method's configuration. 
 * The "set_gd_mini_batch_size" method sets the size of a mini-batch. 
 * The "set_12_coefficient" method sets the value of the regularization weight decay parameter. 
 * The "set_gd_learning_rate" method sets the learning rate parameter. 
 * The "set_gd_monentum" method, we set the maximum number of training epochs, and with the "set_epsilon" method we define the convergence criteria value for a loss function. */
/*The loss function is automatically selected based on the type of labels specified with the "set_labels" method. 
 * We used the "CRegressionLabels" type for the labels because we are solving the regression task. 
 * Network training can be done with the "{train}" method, which takes an object of the "CDenseFeatures" type. 
 * This contains a set of all the training samples.*/

size_t n = 10000; 
...
SGMatrix<float64_t> x_values(1, static_cast<index_t>(n)); 
SGVector<float64_t> y_values(static_cast<index_t>(n)); 
...
auto x = some<CDenseFeatures<float64_t>>(x_values); 
auto y = some<CRegressionLabels>(y_values); 

auto dimensions = x->get_num_features(); 
auto layers = some<CNeuralLayers>(); 
layers = wrap(layers->input(dimensions)); 
layers = wrap(layers->rectified_linear(32)); 
layers = wrap(layers->rectified_linear(16)); 
layers = wrap(layers->rectified_linear(8)); 
layers = wrap(layers->linear(1)); 
auto all_layers = layers->done(); 

auto network = some<CNueuralNetwork>(all_layers); 
network->quick_connect(); 
network->initialize_neural_network(); 

network->set_optimization_method(NNOM_GRADIENT_DESCENT); 
network->set_gd_mini_batch_size(64); 
network->set_l2_coefficient(0.0001); 		/*regularization*/
network->set_max_num_epochs(500); 
network->set_epsilon(0.0); 			/*convergence criteria*/
network->set_gd_learning_rate(0.01); 
network->set_gd_momentum(0.5); 

network->set_labels(y); 
network->train(x); 

/*to see the training process, we can set the higher logging level for the "Shogun" library the calls below*/
shogun::sg_io->set_log_evel(shogun::MSG_DEBUG); 

/*This function allows us to see a lot of additional information about the overall training process, which can help us debug and find problems in the network we train. */
