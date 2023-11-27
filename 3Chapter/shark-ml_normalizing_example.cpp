#include <shark/Algorithms/Trainers/NormalizeComponetsUnitVariance.h>
#include <shark/Models/Normalizer.h>
...
Normalizer<RealVector> normalizer; 
NormalizeComponentsUnitVariance<RealVector> normalizingTrainer(true); 
normalizingTrainer.train(normalizer, dataset.inputs()); 
dataset = transformInputs(dataset, normalizer); 

/*After defining the model and the trainer objects, we have to call the "train()" method to learn statistics from the input dataset, and then we use the "transformInpust()" function to update the target dataset. 
 * We can print a "Shark-ML" dataset with the standard C++ stream operator*/
std::cout << dataset << std::endl; 
