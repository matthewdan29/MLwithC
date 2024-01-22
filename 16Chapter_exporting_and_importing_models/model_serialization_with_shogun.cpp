/*The "Shogun" library can save model parameters in different file formats such as aSCII, JSON, XML, and HDF5. 
 * This library can't load model architectures from a file and is only ablie to save and load the weights of the exact model. 
 * But there is an exveption for neural networks: the "Shogun" library can load a network sturcture from a JSON file.*/

/*Now we start by generating the training data:*/
const int32_t n = 1000; 
SGMatrix<float64_t> x _values(1, n); 
SGVector<float64_t> y_values(n); 

std::random_device rd; 
std::mt19937 re(rd()); 
std::uniform_real_distribution<double> dist(-1.5, 1.5); 

/*generate data*/
/*We filled the 'x' object of the "CDenseFeatures" type with the predictor variable values and the 'y' object of the "CRegressionLabels" type with the target variable values. 
 * THe linear dependence is the same as in the last example.
 * We also rescaled the 'x' values with the object ofthe "CRescaleFeatures" type. */
for (int32_t i = 0; i < n; ++i)
{
	x_values.set_element(i, 0, i); 

	auto y_val = function(i) + dist(re); 
	y_values.set_element(y_val, i); 
}

auto x = some<CDenseFeatures<float64_t>>(x_values); 
auto y = some<CRegressionLeables>(y_values); 

/*rescale*/
auto x_scaler = some<CRescaleFeatures>(); 
x_scaler->fit(x); 
x_scaler->transform(x, true); 


/*To show how the serialization API in the "Shogun" library works, we will use the "CLinearRidgeRegression" and "CNeuralNetwork" models.*/

/*below shows how to train and serialize the "CLinearRidgeRegression" model:*/
void TrainAndSaveLRR(some<CDenseFeatures<float64_t>> x, Some<CRegressionLabels> y)
{
	float64_t tau_regularization = 0.0001; 
	auto model = some<CLinearRidgeRegression>(tau_regularization, nullptr, nullptr); 
	model->set_labels(y); 
	if (!model->train(x))
	{
		std::cerr << "training failed\n"; 
	}

	auto file = some<CSerializeableHdf5File>("sogun-lr.dat", 'w'); 
	if (!model->save_serializable(file))
	{
		std::cerr << "Failed to save the model\n"; 
	}
}

/*The following code shows how to train and save the parameters of a neural network object:*/
void TrainAndSaveNET(Some<CDenseFeatures<float64_t>> x, Some<CRegressionLabels> y)
{
	auto dimensions = x->get_num_features(); 
	auto layers = some<CNeuralLayers>(); 
	layers = wrap(layers->input(dimensions)); 
	layers = wrap(layers->linear(1)); 
	auto all_layers = layers->done(); 

	auto network = some<CNeuralNetwork>(all_layers); 
	/*configure network parameters*/
	...
	
	network->set_labels(y); 
	if (network->train(x))
	{
		auto file = some<CSerializableHdf5File>("shogun-net.dat", 'w');
		if (!network->save_serializable(file))
		{
			std::cerr << "Failed to save the model\n"; 
		}
	} else 
	{
		std:;cerr << "Failed to train the network\n"; 
	}
}

/*Here, we can see the neural network serializatioin is similar to seriablizing the linear model.*/

/*To test our serialized model, we wiil generate a new set of test data.*/
SGMatrix<float64_t> new_x_values(1, 5); 
std::cout << "Target values : \n"; 
for (index_t i = 0; i < 5; ++i)
{
	new_x_values.set_element(static_cast<double>(i), 0, i); 
	std::cout << func(i) << std::endl; 
}

auto new_x = some<CDenseFeatures<float64_t>>(new_x_values); 
x_scaler->transform(new_x, true); 

/*Below is code that deserialization process:*/
void LoadAndPredictLRR(Some<CDenseFeatures<float64_t>> x)
{
	auto file = some<CSerializableHdf5File>("shogun-lr.dat", 'r'); 
	auro model = some<CLinearRidgeRefression>(); 
	if (model->load_serializable(file))
	{
		auto y_predict = model->apply_regression(x); 
		std::cout << "LR predicted values: \n" << y_predict->to_string() << << std::endl; 
	}
}

...
LoadAndPredictLRR(new_x); 

/*There is a particular function that's  used to load a neural network structure from JSON files or strings. 
 * The problem is that we can't export this structure as a file with the library API, so we should make it by ourselves or write our custom exporter. 
 * However, this functionality allows us to define neural network architectures without programming in a declarative style. 
 * This can be useful for experiments becuase we don't need to recompile a whole application. 
 * It also allows us to deploy a new architecture to production without program updates, but note that we need to take care of preserving the input and output network tensor dimensions. */

/*To load the neural network from the JSON-formatted string the "Shogun" library, we can use an object of the "CNeuralNetworkFileReader" type.*/
Some<CNeuralNetwork> NETFromJson()
	/*Here, we defined the neural network architecture with a JSON string.
	 * This is the same neural network architecture that we used for training.*/
{
	CNeuralNetworkFileReader reader; 
	const char* net_str = 
		"{"
			"\"optimization_method\": \"NNOM_GRADIENT_DESENT\","
			"\"max_num_epochs\": 100,"
			"\"gd_mini_batch_size\": 0,"
			"\"gd_learning_rate\": 0.01,"
			"\"gd_momentum\": 0.9,"

			"\"layers":"
			"{"
				"\"type\": \"NeuralInputLayer\","
				"\"num_neurons\: 1,"
				"\"start_index\": 0"
				"},"
			"\"linear1\":"
			"{"
				"\"type\": \"NeuralLinearLayer\","
				"\"num_neurons\": 1,"
				"\"inputs\": [\"input1\"]"
				"}"
			"}"
		"}"; 
	auto network = wrap(reader_string(net_str)); 

	return network; 
}

/*Below code shows how to use the "NETFromJson" function to create a network object from the JSON string and initialize it with the serialized parameters:*/
void LoadAndPredictNET(Some<CDenseFeatures<float64_t>> x)
{
	auto file = some<CSerializableHdf5File>("shogun-net.dat", 'r'); 

	auto network = NETFromJson(); 

	if (network->load_serializable(file))
	{
		auto new_x = some<CDenseFeatures<float64_t>>(x); 
		auto y_predict = network->apply_regression(new_x); 
		std::cout << "Network predicted values: \n" << y_predict->to_string() << std::endl; 
	}
}

/*The newly created neural network object is of the "CNeuralNetwork" type. 
 * We used the "load_serializable" method of the new neural network object to load the previously serialized parameters. 
 * Its essential to preserve the same architecture of ML model objects that are used for serialization and deserialization as a different architecture can lead to runtime errors when deserialization is performed.*/
