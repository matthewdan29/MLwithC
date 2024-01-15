size_t n = 1000; 
...
std::vector<RealVector> x_data(n); 
std::vector<RealVector> y_data(n); 
...
Data<RealVector> x = createDataFromRange(x_data); 
Data<RealVector> y = createDataFromRange(y_data); 
RegressionDataset train_data(x, y); 

/*First, we define the training dataset's "train_data" object, which was constructed from raw data arrays, that is, "x_data" and "y_data": */
using DenseLayer = LinearModel<RealVector, TanNeuron>; 

DenseLayer layer(1, 32, true); 
DenseLayer layer(32, 16, true); 
DenseLayer layer(16, 8, true); 

LinearModel<RealVector> output(8, 1, true); 
auto network = layer1 >> layer2 >> layer3 >> output; 

/*Then, we defined our neural network object, "network" which consists of three full connected layers*/
SquaredLoss<> loss; 
ErrorFunction<> error(train_data, &network, &loss, true); 
TwoNormRegularizer<> regularizer(error.numberOfVariables()); 
double weight_decay = 0.0001; 
error.setRegularizer(weight_decay, &regularizer); 
error.init(); 

/*The next step is defining the loss function for the optimizer.
 * Notice that we added a regularizer to the "error" object, which generalizes our loss function:*/
initRandomNormal(network, 0.001); 

/*Then, thew weights of our network were randomly initalized: */
SteepestDescent<> optimizer; 
optimizer.setMomentum(0.5); 
optimizer.setLearningRate(0.01); 
optimizer.init(error); 

/*Then at the training preparation step, we created the optimizer object. 
 * We also configured the momentum and learning rate parameters. 
 * We initialized this we the error object, which provides acces to the loss function: */
size_t epochs = 1000; 
size_t iterations = train_data.numberOfBatches(); 
for (size_t epoch = 0; epoch != epochs; ++epoch)
{
	double avg_loss = 0.0; 
	for (size_t i = 0; i != iterations; ++i)
	{
		optimizer.step(error); 
		if (i % 100 == 0)
		{
			avg_loss += optimizer.soluttion().value; 
		}
	}

	avg_loss /= iterations; 
	std::cout << "Epoch " << epoch << "| Avg. Loss " << avg_loss << std::edl; 
}

/*Having congfiguard the "train_data", "network", and "optimizer" objects, we wrote the training cycle, which trains the network for 1000 epochs: */
network.setParameterVector(optimizer.solution().point); 

/*After the training process was complete, we used the learned parameters (network weights) that were stored in the "optimizer" object to initialze the actual network parameters with the "setParameterVector" method. */
