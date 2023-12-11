UnlabeledData<RealVector> data; 
importCSV(data, dataset_name); 

/*separate last two samples in test dataset*/
data.spletBatach(0, 50); 
auto test_data = data.splice(1); 

double gamma = 0.5;			/*Kernel bandwidth parameter*/
GaussianRbfKernel<> kernel(gamma); 
KernelExpansion<RealVector> ke(&kernel); 

double nu = 0.5; 			/*Parameter of the method for controlling*/
					/*smoothness of the solution*/
OneClassSvmTrainer<RealVector> trainer(&kernel, nu); 
trainer.stoppingCondition().minAccuracy = 1e-6; 
trainer.train(ke, data); 

double dist_threshold = -0.2; 
RealVector output; 
auto detect = [&](const UnlabeledData<RealVector>& data)
{
	for (size_t i = 0; i < data.numberOfElements(); ++i)
	{
		ke.eval(data.element(i), output); 
		if (output[0] > dist_threshold)
		{
			/*Do something with anomalies*/
		} else 
		{
			/*Do something with normal*/
		}
	}
}; 

detect(data); 
detect(test_data); 

/*	1) We loaded the object of the "UnlabeledData" class from the CSV file and split it into two parts one for training and one for testing 
 *
 *	2) we declared the kernel object of the "GaussianRbfKernel" type and initialized an object of the "kernelExpansion" class with it. 
 *	The "KernelExpansion" class implements an affine linear kernel expansion. */
