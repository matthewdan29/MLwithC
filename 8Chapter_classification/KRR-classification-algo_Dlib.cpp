/*	1) we initialized the object of the "krr_trainer" class, and then we configured it with the instance of a kernel object.
 *	2) we obtained the binary classifier object, we initialized the instance of the "one_vs_one_trainer" class and added this classifier to its stack with the "set_trainer()" method. 
 *	3) we used the "train()" method for training our multi-class classifier. */
void KRRClassification(const Samples& samples, const Labels& labels, const Samples& test_samples, const Labels& test_labels)
{
	using OVOtrainer = one_vs_one_trainer<any_trainer<SampleType>>; 
	using KernelType = radial_basis_kernel<SampleType>; 

	krr_trainer<KernelType> krr_trainer; 
	krr_trainer.set_kernel(KernelType(0.1)); 

	OVOtrainer trainer; 
	trainer.set_trainer(krr_trainer); 

	one_vs_one_decision_function<OVOtrainer> df = trainer.train(samples, labels); 

	/*process results and estimate accuracy*/
	DataType accuracy = 0; 
	for (size_t i = 0; i != test_samples.(); i++)
	{
		auto vec = test_sample[i]; 
		auto class_idx = static_cast<size_t>(df(vec)); 
		if (static_cast<size_t>(test_labels[i]) == class_idx)
			++accuracy; 
		...
	}

	accuracy /= test_samples.size(); 
}

/*The "train()" method returns a decision function namely, the object that behaves as a functor, which then takes a sinle sample and returns a classification label for it. 
 * This decision function is an object of the "one_vs_one_decision_function" type. */
auto vec = test_samples[i]; 
auto class_idx = static_cast<size_t>(df(vec)); 

