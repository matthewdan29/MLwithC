/*Below introduces a function declaration for this kind of task:*/
void LRClassification(const ClassificationDatasete& train, const ClassificationDataset& test, unsigned int num_classes)
{
	...
}

/*Below is how to configure an object for multi-class classification*/
/* These are the steps on how to configure an object for multi-class classification: 
 * 	1) We define the "ovo" object of the "OneVersusOneClassifier" class, which encapsulates the single multi class classifier. 
 * 	2) We initialized all binary classifiers for the one-versus-one strategy and placed them in the "lr" container object of the "std::vector<LinearClassifier<RealVector>>" type. 
 * 	3) We then trained the set of binary classifiers with the trainer object of the "LogisticRegression" type and put them into the "lr" container. 
 * 	4) We then ran the training with nested cycles over all classes. The "ovo_classifiers" object contains the pointers to binary classifiers. These classifiers are configured in such a way that each of them classifies a single class as positive, and all other classes are treated as negative
 * 	5) We then use the "ovo_classifiers" object to populate the "ovo" object, using the "addClass" method*/
OneVersusOneClassifier<RealVector> ovo; 
unsigned int pairs = num_classes * (num_classes - 1) / 2; 
std::vector<LinearClassifier<RealVector>> lr(pairs); 

for (std::size_t n = 0, cls1 = 1; cls1 < num_classes; cls1++)
{
	using BinaryClassifierType = OneVersusOneClassifier<RealVector>::binary_classifier_type; 
	std::vector<BinaryClassifierType*> ovo_classifiers; 
	for (std::size_t cls2 = 0; cls2 < cls1; cls2++, n++)
	{
		/*get the binary subproblem*/
		ClassificationDataset binary_cls_data = binarySubProblem(train, cls2, cls1); 
		/*train the ninary machine*/
		LogisticRegression<RealVector> trainer; 
		trainer.train(lr[n], binary_cls_data); 
		ovo_classifiers.push_back(&lr[n]); 
	}
	ovo.addClass(ovo_classifiers); 
}

/*After we trained all binary classifiers and configured the "OneVersusOneClassifier" object, we used it for model evaluation on a test set. 
 * This object can be used as a functor to classify the set of test examples, but they need to have the "UnlabeledData" type. */

/*estimate accuracy*/
ZeroOneLoss<unsigned int> loss; 
Data<unsigned int> output = ovo(test.inputs()); 
double accuracy = 1. - loss.eval(test.labels(), output); 

/*process results*/
for (std::size_t i = 0; i != test.numberOfElements(); i++)
{
	auto cluser_idx = output.element(i); 
	auto element = test.inputs().element(i); 
	...
}
