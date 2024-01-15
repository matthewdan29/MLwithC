/*All the structural parts of the neural networks in the PyTorch framework should be derived from the "torch::nn::Module" class. 
 * Below is a part of the header files*/

/*"LenNet5Impl" class has a intermediated implementation because PyTorch uses a memory management model based on smart pointers, and all the modules should be wrapped in a special type. */
#include <torch/torch.h>

class LeNet5Impl : public torch::nn::Module
{
	public: 
	LeNet5Impl(); 
	
	/*"forward function takes the network's input and passes it through all the network layers until an output value is returned from this function. "*/
	torch::Tensor forward(torch::Tensor x); 

	private: 
	/*"torch::nn::Sequential" class is used to group sequential layers in the network and automate the process of forwarding values between them. */
	torch::nn::Sequential conv_; 	/*this contains convolutional layers*/
	torch::nn::Sequential full_; 	/*this contains the final fully connected layers*/


}; 
TORCH_MODULE(LeNet5); 	/*"TORCH_MODULE" is a special macro that can do this definition for us auto maticlly; we need to specify the name of our module in order to use it with is "LeNet5"*/

/* "torch::nn::ModuleHolder" is a wrapper around "std::shared_ptr", but also defines some additional methods for managing modules. 
 * If we want to follow all PyTorch convertions an duse our module (network) with all PyTorch's functions without any problems, our module class definition should be like the example below*/
class Name : public torch::nn::ModuleHolder<Impl> {}	/*"Impl" is the implementation of our module, which is derived from the "torch::nn::Module" class*/

/*The PyTorch framework contains many functions for creating layers.
 * The "torch::nn::Conv2d" function created the two dimenstional convolution layer. 
 * Another way to create a layer in PyTorch is to use the "torch::nn::Functional" function to wrap some simple function into the layer, which can then be connected with all the outputs of the previous layer. 
 * Notice that activation functions are not part of the neurons in Pytorch and shold be connected as a separated layer. 
 * Below shows the componets:*/
static std::vector<int64_t> k_size = {2, 2}; 
static std::vector<int64_t> p_size = {0, 0}; 

LeNet5Impl::LeNet5Impl() 
{
	conv_ = torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5)), torch::nn::Functional(torch::tanh), torch::nn::Functional(torch::avg_pool2d, /*kernel_size*/ torch::IntArrayRef(k_size), /*stride*/ torch::InArrayRef(k_size), /*padding*/ torch::IntArrayRef(p_size), /*ceil_mode*/ false, /*count_inclue_pade*/ false), torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5)), torch::nn::Functional(torch::tanh), torch::nn::Functional(torch::avg_pool2d, /*kernel_size*/ torch::IntArrayRef(k_size), /*stride*/ torch::IntArrayRef(k_size), /*pading*/ torch::IntArrayRef(p_size), /*ceil_mode*/ false, /*count_include_pad*/ false), torch::nn::Conv2d(torch::nn::conv2dOptions(16, 120, 5)), torch::nn::Functional(torch::tanh)); 
	register_module("conv", conv_); 

	full_ = torch::nn::Sequential(torch::nn::Linear(torch::nn::LinearOptions(120, 84)), torch::nn::Functional(torch::tanh), torch::nn::Linear(torch::nn::LinearOptions(84, 10))); 
	register_module("full", full_); 
}

/*Here, we initialized two "torch::nn::Sequential" modules. 
 * They take a variale number of other modules as arguments for constructors. 
 * Notice that for the initialization of the "torch::nn::Conv2d" module, we have to pass the instance of the "torch::nn::Conv2dOptions" class, which can be initialized with the number of input channels, the number of output channels, and the kernel size. 
 * We used "torch::tanh" as an activation function; notice that it is wrapped in the "torch::nn::Functional" class instance. 
 * The average pooling function is also function. 
 * Also, the pooling function takes several arguments, so we bound their fixed vaules. 
 * When a function in PyTorch requires the vauesof the dimensions, it assumes that we provide an instance of the "torch::InArrayRef" type. 
 * An object of this type behaves as a wrapper for an array with dimension values. 
 * We should be cardful here because such an array should exits at the same time as the wrapper lifetime; notice that "torch::nn::Fucntioanl" stores "torch::IntArrayRef" objects internally. 
 * That is why we defined "k_size" and "p_size" as static global variables. */
/*"register_module" function associates the string name with the module and register it in the interneals of the parent module. 
 * If the module is registered in such a way, we can use a string based parameter search later and automatic module serialization. 
 * "torch::nn::Linear" module defines the fully connected layer and should be initialized with an instance of the "torch::nn::LinearOptions" type, which defines the number of inputs and the number of outputs, that is, a count of the layer's neurons. 
 * Notice that the last layer returns 10 values, not one label, despite only having a single target label. 
 * THis is the standard approach in classification tasks. */

/*The "forward" function is implemented as follows: 
 * 	
 * 	1) Pass the input tensor (image) to the "forward" function of the sequential convolutional group. 
 *
 * 	2) Next, flattened its output with the "view" tensor method because full connected layer assume that the input is flat. 
 * 	The "view" method takes the new dimensions for the tensor and returns a tensor view without exactly copying the data; -1 means that we don't care about the dimension's value and that it can be flattened. 
 *
 * 	3) Then, the flattened output from the convolutional group is passed to the fully connected group. 
 *
 * 	4) Last, we applied the softmax function to the final output. 
 * 	We're unable to wrap "torch::log_softmax" in the "torch::nn::Functional" class instance because of multiple overrides. */

torch::Tensor LeNet5Impl::forward(at::Tensor x)
{
	auto output = conv_->forward(x); 
	output = output.view({x.size(0), -1}); 
	output = full_->forward(output); 
	output = torch::log_softmax(output, -1); 
	return output; 
}

/*This function was chosen because its results can be directly used for the cross entropy loss function, which measures the differnece between two probability distributions.
 * The target distribution can be directly calculatedfrom the target label value we create the 10 value's vector of zeros and put one in the place indexed by the label value. 
 * Now, we have all the required components to train the neural network. */
