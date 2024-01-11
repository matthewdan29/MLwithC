/*below code snippet shows the network definition with the "Dlib" library API*/
using NetworkType = loss_mean_squared<fc<1, htan<fc<8, htan<fc<16, htan<fc<32, input<matix<double>>>>>>>>>>; 

/*The definition can be read in this following order*/

/*1) we start with the input layer*/
input<matrix<double>>

/*2) Then, we added the first hidden layer with 32 neurons*/
fc <32, input<matrix<double>>>

/*3) After, we added the hyperbolic tangent activation function to the first hidden layer*/
htan<fc<32, input<matix<double>>>>

/*4) Next, we added the second hidden layer with 16 neurons and an activation function: */
htan<fc<16, htan<fc<32, input<matrix<double>>>>>>

/*5) Then, we added the third hidden layer with 8 neurons and an activation function*/
htan<fc<8, htan<fc<16, htan<fc<32, input<matrix<double>>>>>>>>

/*6) Then, we added the last output layer with 1 neuron and without an activation function*/
fc<1, htan<fc<8, htan<fc<16, htan<fc<32, input<matrix<double>>>>>>>>>

/*7) last, we finsihed with the loss funcitoin:*/
loss_mean_squared<...>

/*The following snippet shows the complete source code example with a network definition:*/

size_t n = 10000; 
...
std::vector<matrix<double>> x(n); 
std::vector<float> y(n); 
...
using NetworkType = loss_mean_squared<fc<1, htan<fc<8, htan<fc<16, htan<fc<32, input<matrix<double>>>>>>>>>>; 
NetworkType network; 
float weight_decay = 0.0001f; 
float momentum = 0.5f; 
sgd solver(weight_decay, momentum); 
dnn_trainer<NetworkType> trainer(network, solver); 
trainer.set_learning_rate(0.01); 
trainer.set_learning_rate_shrik_factor(1); 	/*disable learning rate changes*/
trainer.set_mini_batch_size(64); 
trainer.set_max_num_epochs(500); 
trainer.be_verbose(); 
trainer.train(x, y); 
network.clean(); 

auto predictions = network(new_x); 


