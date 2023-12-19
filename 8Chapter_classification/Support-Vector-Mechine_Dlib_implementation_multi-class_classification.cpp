void SVMClassification(const Samples& samples, const Labels& labels, const Samples& test_samples, const Labels& test_labels)
{
	using OVOtrainer = one_vs_one_trainer<any_trainer<SampleType>>; 
	using KernelType = radial_basis_kernel<SampleType>; 

	svm_nu_trainer<KernelType> svm_trainer; 
	svm_trainer.set_kernel(KernelType(0.1)); 

	OVOtrainer trainer; 
	trainer.set_trainer(svm_trainer); 

	one_vs_one_decision_function<OVOtrainer> df = trainer.train(samples, labels); 

	/*process results and estimate accuracy*/
	DataType accuracy = 0; 
	for (size_t i = 0; i != test_samples.size(); i++)
	{
		auto vec = test_sample[i]; 
		auto class_idx = static_cast<size_t>(df(vec)); 
		if (static_cast<size_t>(test_labels[i]) == class_idx)
			++accuracy; 
		...
	}

	accuracy /= test_samples.size(); 
}
