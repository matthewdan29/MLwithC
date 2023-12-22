/*First, we need to define weak (or elementary) algo that we are going to use for stacking. 
 * To unify access to the weak algorithms, we defined the base class*/
struct WeakModel
{
	virtual ~WeakModel() {}
	virtual void train(const ClassificationDataset& data_set) = 0; 
	virtual LinearClassifier<RealVector>& GetClassifier() = 0; 	
}; 

/*The logistic regression, the linear discriminant analysis(LDA), and the linear SVM models are used for creating weak algorithms.*/
struct LogisticRegressionModel : public WeakModel
{
	LinearClassifier<RealVector> classifier; 
	LogisticRegression<RealVector> trainer; 
	void Train(const ClassificationDataset& data_set) override
	{
		trainer.train(classifier, data_set); 
	}

	LinearClassifier<RealVector>& getClassifier() override 
	{
	return classifier; 
	}
}; 

struct LDAModel : public WeakModel
{
	LinearClassifier<Realvector> classifier; 
	LDA trainer; 
	void Train(const ClassificationDataset& data_set) override
	{
		trainer.train(classifier, data_set); 
	}
	LinearClassifier<RealVector>& GetClassifier() override 
	{
		return classifier; 
	}
}; 

struct LinearSVMModel : public WeakModel
{
	LinearClassifier<RealVector> classifier; 
	LinearCSvmTrainer<RealVector> trainer{SVM_C, false}; 
	void Train(const ClassificationDataset& data_set) override
	{
		trainer.train(classifier, data_set); 
	}

	LinearClassifier<RealVector>& GetClassifier() override 
	{
		return classifier; 
	}
}; 

/*One of the crucial moments for the stacking approach is combining results of weak algo to one set, which is used for training or evaluating the meta algorithm. 
 * There is the "MakeMetaSet" method in our implementation: 
 * 	1) Takes the vector of predictions from weak algorithms, 
 * 	2) take the vector of corresponding labels from the original dataset
 * 	3) combines them into a new object of the "ClassificationDataset". */

/*This method creates two vector of inputs and labels and uses the "shark-ml" function "CreateLabeledDataFromRange" to create a new dataset.*/

ClassificationDataset MakeMetaSet(const std::vector<Data<unsigned int>>& inputs, const Data<unsigned int>& labels)
{
	auto num_elements = labels.numberOfElements(); 
	std::vector<RealVector> vinputs(num_elements); 
	std::vector<unsigned int> vlabels(num_elements); 
	std::vector<RealVector::value_type> vals(inputs.size()); 
	for (size_t i = 0; i < num_elements; ++i)
	{
		for (size_t j = 0; j < inputs.size(); ++j)
		{
			vals[j] = inputs[j].elements(i); 
		}

		vinputs[i] = RealVector(vals.begin(), vals.end()); 
		vlabels[i] = labels.elements(i); 
	}

	return createLabeledDataFromRange(vinputs, vlabels); 
}

/*Because of the nature of the selected algorithms, we need to normalize our data. 
 * Lets assume we have two datasets for training and testing*/
void StackingEnsemble(const ClassificationDataset& train, const ClassificationDataset& test)
{
	...
}

/*To normalize the training dataset, we need to copy the original dataset because the "Normalizer" algorithm works in place and modifies the objects with which it works.*/
ClassificationDataset train_data_set = train; 
train_data_set.makeIndependent(); 

/*You train the normalizer first, and only then can we apply it to the "transformInputs" function which transform only input features because we don't need to normalize binary labels*/
bool removeMean = true; 
Normalizer<RealVector> normalizer; 
NormalizeComponentsUnitVariance<RealVector> normalizing_trainer(removeMean); 
normalizing_trainer.train(normalizer, train_data_set.inputs()); 
train_data_set = transformInputs(train_data_set, normalizer); 

/*To spped up and generalize the models also reduced the dimensionality of the training features with the "PCA"(Principal component analysis) algorithm */
PCA pca(train_data_set.inputs()); 
LinearModel<> pca_encoder; 
pca.encoder(pca_encoder, 5); 
train_data_set = transformInputs(train_data_set, pca_encoder); 

/*Next, after preprocessing our training dataset, we can define and train the weak models that we are going to use for evaluation.*/
/*weak models*/
std::vector<std::shared_ptr<WeakModel>> weak_models; 
weak_models.push_back(std::make_shared<LogisticRegressionModel>()); 
weak_models.push_back(std::make_shared<LDAModel>()); 
weak_models.push_back(std::make_shared<LinearSVMModel>()); 

/*train weak models for predictions*/
for (auto weak_model : weak_models)
{
	weak_model->Train(train_data_set); 
}

/*For training, the meta-algorithm needs to get the meta-features, and, according to the stacking approach, we will split our training dataset into 10 folds*/
/*There is the "createCVSameSizeBalanced" function, it creates equal-size folds, wehre each consists of two parts: 
 * 	1) the training part 
 * 	2) the validation part. 
 * We wil iterate over created folds to train weak models and create meta-features.*/

size_t num_partitions = 10; 
ClassificationDataset meta_data_train; 
auto folds = createCVSameSizeBalanced(train_data_set, num_partitions); 
for (std::size_t i = 0; i != folds.size(); ++i)
{
	/*access the fold*/
	ClassificationDataset training = folds.training(i); 
	ClassificationDataset validation = folds.validation(i); 

	/*train local weak models - new ones on each of the folds*/
	std::vector<std::shared_ptr<WeakModel>> local_weak_models; 
	local_weak_models.push_back(std::make_shared<LogisticRegressionModel>());
	local_weak_models.push_back(std::make_shared<LDAModel>()); 
	local_weak_models.push_back(std::make_shared<LinearSVMModel>()); 

	std::vector<Data<unsigned int>> meta_predictions; 
	for (auto weak_model : local_weak_models)
	{
		weak_model->Train(training); 
		auto predictions = weak_model->GetClassifier()(validation.inputs()); 
		meta_predictions.push_back(predictions); 
	}

	/*combine meta features*/
	meta_data_train.append(MakeMetaSet(meta_predictions, validation.labels())); 
}

/*The "meta_data_train" object contains the meta-features and is used to train the meta-model, which is the regular linear SVM model in our case*/
LinearClassifier<RealVector> meta_model; 
LinearCSvmTrainer<RealVector> trainer(SVM_C, true); 
trainer.train(meta_model, meta_data_train); 

/*Since we used data preprocessing, we should also transform our test data in the same way that we transformed our training data. 
 * This can be easily done with the "normalizer" and the "pca_encoder" objects, which are already trained and hold the required transformation options inside. 
 * Such objects should be stored on secondary storage.*/

ClassificationDataset test_data_set = test; 
test_data_set.makeIndependent(); 
test_data_set = transformInputs(test_data_set, normalizer); 
test_data_set = transformInputs(test_data_set, pca_encoder); 

/*Now, make the "meta_test" dataset object in the same way as we made the training meta-dataset. 
 * we will store predictions from every weak model in the "meta_predictions" vector and will use our helper function to combine them in the object of the "ClassificationDataset" type*/
std::vector<Data<unsigned int>> meta_predictions; 
for (auto weak_model : weak_models)
{
	auto predictions = weak_model->GetClassifier()(test_data_set.inputs()); 
	meta_predictins.push_back(predictions); 
}

ClassificationDataset meta_test = MakeMetaSet(meta_predictions, test_data_set.label()); 

/*After we have created the meta-features, we can pass them as input to the "meta_model" object to generate the real predictions. 
 * Also the accuracy is calcuated like below*/
Data<unsigned int> predictions = meta_model(meta_test.inputs()); 

ZeroOneLoss<unsigned int> loss; 
double accuracy = 1. - loss.eval(meta_test.labels(), predictions); 
std::cout << "Stacking ensemble accuracy = " << accuracy << std::endl; 
