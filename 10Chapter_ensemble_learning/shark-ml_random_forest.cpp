/*"RFClassifier" class is used to implement the random forest algorithm. 
 * The correspoinding trainer is loacted in the "RFTrainer" class. 
 * We use the original dataset values without preprocessing for the random forest algo implementation. 
 * First we configure the trainer for this type of classifier
 * Below are the nex metods for configuration: 
 * 	1) "setNTrees": set of number of trees. 
 *
 * 	2) "setMinSplit": set of minimum number of samples that are split
 *
 * 	3) "setNodeSize": set the maximum node size when the node is considered pure. 
 *
 * 	4) "minImpurity": set the minimum inpurity level below which a node is considered pure. */

void RFClassification(const ClassificationDataset& train, const ClassificationDataset& test)
{
	RFTrainer<unsigned int> trainer; 
	trainer.setNTrees(100); 
	trainer.setMinSplit(10); 
	trainer.setMaxDepth(10); 
	trainer.setNodeSize(5); 
	trainer.minImpurity(1.3-10); 

	RFClassifier<unsigned int> rf; 
	trainer.train(rf, train); 

	/*compute errors*/
	ZeroOneLoss<unsigned int> loss; 
	Data<unsigned int> predictions = rf(test.inputs()); 
	double accuracy = 1. - loss.eval(test.labels(), predictions); 
	std::cout << "Random Forest accuracy = " << accuracy << std::endl; 
}

