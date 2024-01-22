/*The "Dlib" library uses the serialization API for "decision_function" and neural network type objects.*/

/*First, we define the types for the neural network, regression kernel, and training sample:*/
using namespace dlib; 

using NetworkType = loss_mean_squared<fc<1, input<matrix<double>>>>; 
using SampleType = matrix<double, 1, 1>; 
using KernelType = linear_kernel<SampleType>; 

/*Then, we generate the training data:*/
size_t n = 1000; 
std::vector<matrix<double>> x(n); 
std::vector<float> y(n); 

std::random_device rd; 
std::mt19937 re(rd()); 
std::uniform_real_distribution<float> dist(-1.5, 1.5); 

/*generate data*/
for (size_t i = 0; i < n; ++i)
{
	x[i](0, 0) = i; 
	y[i] = func(i) + dist(re); 
}

/*'x' represents the predictor varibale, while 'y' is the target variable. 
 * The target variable, 'y' is salted with uniform random noise to simulate real data. 
 * These variables have a linear dependency, which is defined below:*/
double func(double x)
{
	return 4. + 0.3 * x; 
}

/*After we have generated the data, we normalize it using the "vector_normalizer" type object. 
 * Objects of this type can be resued after training to normalize data with the learned mean and st. Dev.*/
vector_normalizer<matrix<double>> normalizer_x; 
normalizer_x.train(x); 

for (size_t i = 0; i < x.size(); ++i)
{
	x[i] = normalizer_x(x[i]); 
}

/*Next, we train the "decision_function" object for kernel ridge regression with the "krr_trainer" type object: */
void TrainSAndSaveKRR(const std::vector<matix<double>>& x, const std::vector<float>& y)
{
	krr_trainer<KernelType> trainer; 
	trainer.set_kernel(KernelType()); 
	decision_function<KernelType> df = trainer.train(x, y); 
	serialize("dlib-krr.dat") << df; 
}

/*Now that we have the trained "decision_function" object, we can serialize it into a file with a stream object that's returned by the "serialize" function:*/
serialize("dlib-krr.dat") << df; 
/*aout function takes the name of the file for storage and returns an output stream object. 
 * We used the "<<" operator to put the learned weights of the regresson model into the file. 
 * This serialization approach only saves the model parameters*/

/*The same approach can be used to serialize almost all ML models in the "Dlib" library.*/
/*For neural networks, there is also the "net_to_xml" function, which saves the model structure, but there is no function to load this saved structure into our program. 
 * It is the user's responsibility to implement a loading function. 
 * The "net_to_xml" function exists if we wish to share the model between frameworks as it is written in the "Dlib" documentation.*/
/* below shows how to use it to serialize the parameters of a neural network:*/
void TrainAndSaveNetwork(const std::vector<matrix<double>>& x, const std::vector<float>& y)
{
	NetworkType network; 
	sgd solver; 
	dnn_trainer<NetworkType> trainer(network, solver); 
	trainer.set_learning_rate(0.0001); 
	trainer.set_mini_batch_size(50); 
	trainer.set_max_num_epochs(300); 
	trainer.be_verbose(); 
	trainer.train(x, y); 
	network.clean(); 

	serialize("dlib-net.dat") << network; 
	net_to_xml(network, "net.xml"); 
}

/*To check that parameter serialization works as expected, we generate new test data to evaluate a loaded model on them:*/
std::cout << "Target values \n"; 
std::vector<matrix<double>> new_x(5); 
for (size_t i = 0; i < 5; ++i)
{
	new_x[i].set_size(1, 1); 
	new_x[i](0, 0) = i; 
	new_x[i] = normalizer_x(new_x[i]); 
	std::cout << func(i) << std::endl;
}

/*To load a serialized object in the "Dlib" library, we can use the "deserialize" function. 
 * This function takes the file name and returns the input stream object:*/
void LoadAndPredictKRR(const std::vector<matrix<double>>& x)
{
	decision_function<KernelType> df; 
	deserialize("dlib-krr.dat") >> df; 

	/*Predict*/
	std::cout << "KRR predictions \n"; 
	for (auto& v : x)
	{
		auto p = df(v); 
		std::cout << static_cast<double>(p) << std::endl; 
	}
}

/*As we discussed, serialization in the "Dlib" library only stores model parameters. 
 * So, to load them, we need to use the model object with the same properties that it hade before serialization was performed. 
 * For a regression model, this means that we should instantiate a decision function object with the same kernel type. 
 * For a neural network model, this means that we should instantiate a network object ofthe same type that we used for serialization:*/
void LoadAndPredictNetwork(const std::vector<matrix<double>>& x)
{
	NetworkType network; 
	deserialize("dlib-net.dat") >> network; 

	/*Predict*/
	auto predictions = network(x); 
	std::cout << "Net predictions \n"; 
	for (auto p : predictions)
	{
		std::cout << static_cast<double>(p) << std::endl; 
	}
}


