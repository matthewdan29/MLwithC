/*First, we should define all the layers and connect them in the network. 
The layer can be defined as an object of the "LinearModel" class, parameterized with a specific activation function type. 
In our case, the type of the layer is: */
using DenseLayer = LinearModel<RealVector, TanNeuron>; 

/*To instantiate objects ofthis type, we have to pass three arguments to the constructor: 
 * 		1) The number of inputs 
 *
 * 		2) the numeber of neurons(outputs)
 *
 * 		3) The Boolean value that enables a bais if it is equal to true.
 * the ">>" operator can be used to connect all the layer in the network:*/
auto network = layer1 >> layer2 >> layer3 >> output; 



