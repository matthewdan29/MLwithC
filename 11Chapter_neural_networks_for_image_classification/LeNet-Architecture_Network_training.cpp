/*First, we should create PyTorch data loader objects for the train and test dataset. 
 * The data loader object is reponsible for sampling objects from the dataset and making mini batch from them. 
 * This object can be configured as followed: 
 *
 * 	1) First, we intialize the "MNISTDataset" type objects representing our datasets. 
 *
 * 	2) Then, we use the "torch::data::make_data_loader" function to create a data loader object. 
 * 	This function takes the "torch::data::DataLoaderOptions" type object with configuration settings for the data loader. 
 * 	We set the min batch size equal to 256 items and set 8 parallel data loading threads. 
 * 	We should also configure the sampler type, but in the case, we'll leave the default one -- the random sampler. */

/*below show how to initalize the train and test data loaders*/

auto train_images = root_path / "train-images-idx3-ubyte"; 
auto train_labels = root_path / "train-labels-idx1-ubyte"; 
auto test_images = root_path / "t10k-images-idx3-ubyte"; 
auto test_labels = root_path / "t10k-labels-idx1-ubyte"; 

/*initialize train dataset*/
/*-----------------------------------------*/
MNISTDataset train_dataset(train_images.native(), train_labels.native()); 

auto train_loader = torch::data::make_data_loader(train_dataset.map(torch::data::transforms::Stack<>()), torch::data::DataLoaderOptions().batch_size(256).workers(8)); 

/*initialize test dataset*/
/*--------------------------------------------------------*/
MNISTDataset test_dataset(test_images.native(), test_labels.native()); 

auto test_loader = torch::data::make_data_loader(test_dataset.map(torch::data::transforms::Stack<>()), torch::data::DataLoaderOptions().batch_size(1024).workers(8)); 

/*Notice that we didn't pass our dataset objects directly to the "torch::data::make_data_loader" function, but we applied the stacking transformation mapping to it. 
 * This tranformation allows us to sample mini batches in the form fo the "torch::Tensor" object. 
 * If we skip this tranformation, the mini-batches will be sampled as the C++ vectors of tensors. 
 * Ussually, this isn't very usful because we can't apply linear algebra operations to the whole batch in a vectorized manner. */

/*Next, initialze the neural network object of the "LeNet5" type, which we defined previously. 
 * We'll move it ot the GPU to improve training and evaluation performance:*/
LeNet5 model; 
model->to(torch::DeviceType::CUDA); 

/*When the model of our neural network has been initialized, we can initialze an optimizer. 
 * We chose stochastic gradient descent with momentum optimization for this. 
 * it is implemented in the "torch::optim::SGD" class. 
 * The object of this class should be initialized with model (network) parameters and the "torch::optim::SGDOptions" type object. 
 * All "torch::nn::Modue" type objects have the "parameters()" method, which returns the "std::vector<Tensor>" object containing all the parameters (weights) of the network. 
 * There is also the "named_parameters" methods, which returns the dictioinary of named parameters. 
 * Parameter names are created with the names we used in the "register_module" function call. 
 * This  method is hadny if we want to filter parameters and exclude some of them from the training process.*/

/*"torch::optim::SGDOptions" object can be configured with the values of the learning rate, the weight decay regularization factor, and the momentum value factor:*/

double learning_rate = 0.01; 
double weight_decay = 0.0001; 		/*regularization parameter*/
torch::optim::SGD  optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate).weight_decay(weight_decay).momentum(0.5)); 

/*Now that we have our initialzed data loaders, the "network" object, and the "optimizer" object, we are ready to start the training cycle, below shows the training cycle's implementation: */
int epochs = 100; 
for (int epoch = 0; epoch < epochs; ++epoch)
{
	model->train(); 		/*switch to the training mode*/

	/*Iterate the data loader to get batches from the dataset*/
	int batch_index = 0; 
	for (auto& batch : (*train_loader))
	{
		/*Clear gradients*/
		optimizer.zero_grad(); 

		/*Execute the model on the input data*/
		torch::Tensor prediction = model->forward(batch.data); 

		/*Compute a loss value to estimate error of our model */
		/*target should have size of [batch_size]*/
		torch::Tensor loss = torch::nll_loss(prediction, batch.target.squeeze(1)); 

		/*Compute gradients of the loss and parameters of our model*/
		loss.backward(); 

		/*update the parameters based on the calculated gradients*/
		optimizer.step(); 

		/*Output the loss every 10 batches.*/
		if (++batch_index % 10 == 0)
		{
			std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.intem<float>() << std::endl; 
		}
	}

	/*We've made a loop that repeats the training cycle for 100 epoches. 
	 * At the begining of the training cycle, we switched our network object to training mode with "model->train()". 
	 * For one epoch, we itrate over all the mini batches provided by the data loader object: */

	for (auto& batch : (*train_loader))
	{
		...
	}

	/*For every mini-batch we did the next training steps, cleared the previous gradient values by calling the "zero_grad" method for the optimzer object, made a forward step over the network object, "model->forward(batch.data)", and computed the loss value with the "nll_loss" function. 
	 * This function computes the negative log linkihood loss. 
	 * it takes two parameters: 
	 * 	1) The vector containing the probability that a trainig sample belongs to a class identifed by position in the vector 
	 *
	 * 	2) The numeric class label (number). 
	 * Then we call the "backward" method of the optimizer object, which updated all the parameters (weights) and their corresponding gradient values. 
	 * The "step" method only updated the parameters that werer used for initialization. */

	/*Its common practice to use test or validation data to check the training process after each epoch. */
	model->eval(); 		/*switch to the training mode*/
	unsigned long total_correct = 0; 
	float avg_loss = 0.0; 
	for (auto& batch : (*test_loader))
	{
		/*Execute the model on the input data*/
		torch::Tensor prediction = model->forward(batch.data); 

		/*Compute a loss value to estimate error of our model*/
		torch::Tensor loss = torch::nll_loss(prediction, batch.target.squeeze(1)); 

		avg_loss += loss.sum().item<float>(); 
		auto pred = std::get<1>(prediction.detach_().max(1)); 
		total_correct += static_cast<unsigned long>(pred.eq(batch.target.view_as(pred)).sum().item<long>()); 
	}

	avg_loss /= test_dataset.size().value(); 
	double accuracy = (static_cast<double>(total_correct) / test_dataset.size().value()); 
	std::cout << "Test Avg. Loss; " << avg_loss << " | Accuracy: " << accuracy << std::endl; 

	/*Next, we calculate the accuracy value. 
	 * This is the ratio between correct answers and misclassified ones. 
	 * First, we determin the predicted class labels by using the "max" method of the tensor object: */
	auto pred = std::get<1>(prediction.detach_().max(1)); 

	/*The "max" method returns a tuple, where the values are the maximum value of each row of the input tensor in the gibven dimension and the location indices of each maximum value the method found. 
	 * Then, we compare the predicteed labels with the target ones and calculate the number of correct answers:*/
	total_correct+= static_cast<unsigned long>(pred.eq(batch.target.view_as(pred)).sum().item<long>()); 

	/*We use the "eq" tensor's method for our comparison. 
	 * This method returns a boolean vector whose size is equal to the input vector, with values equal to 1 where the vector element compnets are equal and with values equal to 0 where they're not. 
	 * To perform the comparison operation, we made a view for the target labels tensor with the sam dimensions as the predictions tensor. 
	 * The "view_as" method is used for this comparsion. 
	 * Then, we calculated the sum of 1's and moved the value to the CPU variable with the "item<long>()" method. */
}
