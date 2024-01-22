/* The second apprach is to use an object of the "torch::serialize::OutputArchive" type and write the parameters we want to save into it. 
 * Below code shows how to implement the "SaveWeights" method for our model. 
 * This method writes all the parameters and buffers that exist in our module to the "archive" object, and then it uses the "save_to" method to write them in a file:*/
void NetImpl::SaveWeights(const std::string& file_name)
{
	torch::serialize::OutputArchive archive; 
	auto parameters = named_parameters(true); 
	auto buffers = named_buffers(true);		/*Buffers can be retrieved from a module with the "named_buffers" module's method.
	These objects represent the inermediate values that are used to evalueate defferent modules.*/
	for (const auto& param : parameters)
	{
		if (param.value().defined())
		{
			archive.write(param.key(), param.value()); 
		}
	}

	for (const auto& buffer : buffers)
	{
		if (buffer.value().defined())
		{
			archive.write(buffer.key(), buffer.value(), true); 
		}
	}

	archive.save_to(file_name); 
}

/*To load parameters that have been saved this way, we can use the "torch::serialize::InputArchive" object. 
 * Below code shows how to implement the "LoadWeights" method for our model:*/
void NetImpl::LoadWeights(const std::string& file_name)
{
	torch::serialize::InputArchive archive; 
	archive.load_from(file_name); 
	torch::NoGradGuard no_grad; 
	auto parameters = named_parameters(true); 
	auto buffers = named_buffers(true); 
	for (auto& param : parameters)
	{
		archive.read(param.key(), param.value()); 
	}

	for (auto& buffer : buffers)
	{
		archive.read(buffer.key(), buffer.value(), true); 
	}
}
/*above "NetImpl::LoadWeights()" object uses the "load_from" method of the "archive" object to load parameters form the file. 
 * Then, we took the parameters and buffers from our module with the "named_parameters" and "named_buffers" methods and incrementally filled in their values with the "read" method of the "archive" object.*/

/*Now, we can use the new instance of our "model_loaded" model with load parameters to evaluate the model on some test data. 
 * Note that we need to switch the model to the evaluation model with the "eval" method. 
 * Generated test data values should also be converted into tensors objects with the "torch::tensor" function and move to the same computational device that our model uses. */
model_load->to(device); 
model_load->eval(); 
std::cout << "Test:\n"; 
for (int i = 0; i < 5; ++i)
{
	auto x_val = static_cast<float>(i) + 0.1f; 
	auto tx = torch::tensor(x_val, torch::dtype(torch::kFloat).device(device)); 
	tx = (tx - x_mean) / x_std; 

	auto ty = torch::tensor(func(x_val), torch::dtype(torch::kFloat).device(device)); 
	torch::Tensor prediction = model_loaded->forward(tx); 

	std::cout << "Target:" << ty << std::endl; 
	std::cout << "Prediction:" << predction << std::endl; 
}
