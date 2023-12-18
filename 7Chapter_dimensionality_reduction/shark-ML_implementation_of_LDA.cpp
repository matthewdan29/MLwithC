/*Frist, we have to train the algorithm with the "train()" method, which take two parameters: 
 * 		1) First one is reference to the object of the "LinearClassifier" class
 * 		2) second is the dataset reference. 
 * 
 * After the object of the "LinearClassifier" class has been trained, we can use it for data classification as the functional object. 
 * For dimensionality reduction, we have to use the decision function for data transformation. 
 * This function can be retrieved using the "decisionFunction()" method of the "LinearClassifier" class. 
 * The decision function object can be used to transform the input data into a new projetion that can be obtained with the LDA. 
 * After we have the new labels and projected data, we can use them to obtain dimensionality reduced data. */
void LDAReduction(const UnlabeledData<RealVector> &data, const UnlabeledData<RealVector> &labels, size_t target_dim)
{
	LinearClassifier<> encoder; 
	LDA lda; 

	labeledData<RealVector, unsigned int> dataset( labels.numberOfElements(), InputLabelPair<RealVectro, unsigned int>( RealVector(data.element(0).size()), 0)); 
	for (size_t i = 0; i < labels.numberOfElements(); ++i)
	{
		/*labels should start from 0*/
		dataset.element(i).label = static_cast<unsigned int>(labels.element(i)[0]) - 1; 
		data.element(i).input = data.element(i); 
	}
		lda.train(endcoder, dataset); 

		/*project data*/
		auto new_labels = encoder(data); 
		auto dc = encoder.decisionFunction(); 
		auto new_data = dc(data); 
		
		for (size_t i = 0; i < new_data.numberOfElements(); ++i)
		{
			auto l = new_labels.elements(i); 
			auto x = new_data.element(i)[l]; 
			auto y = new_data.element(i)[l]; 
		}
}
