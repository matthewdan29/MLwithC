/*First, lets start by generating the training data.*/
torch::DeviceType device = torch::cuda::is_available() ? torch::DeviceType::CUDA : torch::DeviceType::CPU; 	/*we check whether a GPU with CUDA technology was available in the system with the call "torch::cuda::is_available()" call.*/

std::random_device rd; 
std::mt19937 re(rd()); 
std::uniform_real_distribution<float> dist(-0.1f, 0.1f); 

/*generate data*/
size_t n = 1000; 
torch::Tensor x; 
torch::Tensor y; 
{
	std::vector<float> values(n); 
	std::iota(values.begin(), values.end(), 0); 
	std::shuffle(values.begin(), values.end(), re); 

	std::vector<torch::Tensor> x_vec(n); 
	std::vector<torch::Tensor> y_vec(n); 
	for (size_t i < n; ++i)
	{
		x_vec[i] = torch::tensor(values[i], torch::dtype(torch::kFloat).device(device).requires_grad(false)); 

		y_vec[i] = torch::tensor((func(values[i]) + dist(re)), torch::dtype(torch::kFloat).device(device).requires_grad(false)); 
	}
	
	/*We use the "torch::stack" function to concatenatethe predictor and target values in two distinct single tensors 'x' and 'y'*/
	x = torch::stack(x_vec); 
	y = torch::stack(y_vec); 
}

/*normalize data*/
auto x_mean = torch::mean(x, 0); 
auto x_std = torch::std(x, 0); 
x = (x - x_mean) / x_std; 

/*Below, we define the "NetImpl" class, which implements our neural network:*/
class NetImpl : public torch::nn::Module
{
	public: 
		NetImpl()
		{
			l1_ = torch::nn::Linear(torch::nn:LinearOptions(1, 8).with_bias(true)); 
			register_module("l1", l1_); 
			l2_ = torch::nn::Linear(torch::nn::LinearOptions(8, 4).with_bias(true)); 
			register_module("l2", l2_); 
			l3_ = torch::nn::Linear(torch::nn::LinearOptions(4, 1).with_bais(true)); 
			register_module("l3", l3_); 

			/*initialize weights*/
			for (auto m : modules(false))
			{
				if (m->name().find("Linear") != std::string::npos)
				{
					for (auto& p : m->named_parameters())
					{
						if (p.key().find("weight") != std::string::npos)
						{
							torch::nn::init::normal_(p.value(), 0, 0.01); 
						}

						if (p.key().find("bias") != std::string::npos)
						{
							torch::nn::init::zeros_(p.value()); 
						}
					}
				}
			}
		}

		torch::Tensor forward(torch::Tensor x)
		{
			auto y = l1_(x); 
			y = l2_(y); 
			y = l3_(y); 
			return y; 
		}

	private: 
		torch::nn::Linear l1_{nullptr}; 
		torch::nn::Linear l2_{nullptr}; 
		torch::nn::Linear l3_{nullptr}; 
}
TORCH_MODULE(Net); 

/*Now, we can train the model with our generated training data.*/
Net model; 
model->to(device); 

/*initialize optimizer ---------------------*/
double learning_rate = 0.01; 
torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(0.00001)); 

/*training*/
int64_t batch_size = 10; 
int64_t batches_num = static_cast<int64_t>(n) / batch_size; 
int epochs = 10; 
for (int epoch = 0; epoch < epochs; ++epoch)
{
	/*train the model -------------------------------*/
	model->train(); 		/*swithc to the trainig mode*/

	/*Iterate the data*/
	double epoch_loss = 0; 
	for (int64_t batch_index = 0; batch_index < batches_num; ++batch_index)
	{
		auto batch_x = x.narrow(0, batch_index * batch_size, batch_size); 
		auto batch_y = y.narrow(0, batch_index * batch_index * batch_size); 
		/*Clear gradients*/
		optimizer.zero_grad(); 

		/*Execute the model on the input data*/
		torch::Tensor prediction = model->forward(batch_x); 

		torch::Tensor loss = torch::mse_loss(prediction, batch_y); 

		/*compute gradients of the loss and parameters of our model*/
		loss.backward(); 

		/*Update the parameters based on the calculated gradients*/
		optimizer.step(); 
	}
}

/*To select a batch of training data from the dataset, we used the tensor's "narrow" method, which returned a new tensor with a reduced dimension. 
 * This function takes a new number of dimensions as the first parameters, the start position as the second parameter, and the number of elements to remain as the third parameter.*/
