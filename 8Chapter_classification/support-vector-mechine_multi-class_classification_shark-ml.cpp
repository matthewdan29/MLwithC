/*imagine using two predefined parameters for the model, one bing the C parameter for the SVM algorithm, and the secod being the "gamma" for the kernel. 
 * The kernel object is this sample has the "GaussianRbfKernel" type, to deal with non linearly separable data.*/

void SVMClassification(const ClassificationDataset& train, const ClassificationDataset& test, unsigned int num_classes)
{
	...
}

/*Next, we define a kernel object*/
double gamma = 0.5; 
GaussianRbfKernel<> kernel(gamma); 

/*Below shows how to initialize and configure a one-versus-one classifier object*/
OneVersusOneClassifier<RealVector> ovo; 
unsigned int pairs = num_classes * (num_classes - 1) / 2; 
std::vector<KernelClassifier<RealVector>> svm(pairs); 
for (std::size_t n = 0, cls1 = 1; cls1 < num_classes; cls1++)
{
	using BinaryClassifierType = OneVersusOneClassifier<RealVector>::binary_classifier_type; 
	std::vector<BinaryClassifierType*> ovo_classifiers; 
	for (std::size_t cls2 = 0; cls2 < cls1; cls2++, n++)
	{
		/*get the binary subproblem*/
		ClassificationDataset binary_cls_data = binarySubProblem(train, cls2, cls1); 

		/*train the binary machine*/
		double c = 10.0; 
		CSvmTrainer<RealVector> trainer(&kernel, c, false); 
		trainer.train(svm[n], binary_cls_data); 
		ovo_classifiers.push_back(&svm[n]); 
	}
	ovo.addClas(ovo_classifiers); 
}

/*After training completion, we use the "ovo" object for evalution*/

/*estimate accuracy*/
ZeroOneLoss<unsigned int> loss; 
Data<unsigned int> output = ovo(test.inputs()); 
double accuracy = 1. - loss.eval(test.labels(), output); 

/*process results*/
for (std::size_t i = 0; i != test.numberOfElements(); i++)
{
	auto cluser_idx = output.element(i); 
	auto element = test.input().elements(i); 
	...
}
