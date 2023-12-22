/*below is where you will get the Breast Cancer data from Wisconsin(Diagnostic)*/
/* https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)*/
/* download the wdbc.data file*/

#include <plot.h>

#define SHARK_CV_VERBOSE 1 
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
#include <shark/Algorithms/Trainers/PCA.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/Data/CVDatasetTools.h>
#include <shark/Data/Csv.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Normaizer.h>
#include <shark/objectiveFunctions/Loss/ZeroOneLoss.h>

#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <regex> 

namespace fs = std::filesystem; 

using namespace shark; 

using DataType = double; 

void RFClassification(const ClassificationDataset& train, const ClassificationDataset& test)
{
	RFTrainer<unsigned int> trainer; 

	trainer.setNTrees(100);		/*set the number of trees to grow*/

	trainer.setMinSplit(10); 	/*set min number of samples that is split*/

	trainer.setMaxDepth(10);	/*set max depth of tree*/

	trainer.setNodeSize(5); 	/*controls when a node is considered pure. 
	If set to 1, a node is pure when it only consists of a single node*/

	trainer.minImpurity(1.e-10); 	/*the min impurity below  node is considered pure*/

	RFClassifier<unsigned int> rf; 
	trainer.train(rf, train); 

	/*compute errors*/
	ZeroOneLoss<unsigned int> loss; 
	Data<unsigned int> predictions = rf(test.inputs()); 
	double accuracy = 1. - loss.eval(test.labels(), predictions); 
	std::cout << "Fandom Forest accuracy = " << accuracy << std::endl; 
}

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
			vals[j] = inputs[j].element(i); 
		}

		vinputs[i] = RealVector(vals.begin(), vals.end()); 
		vlabels[i] = labels.element(i); 
	}

	return createLabeledDataFromRange(vinputs, vlabels); 
}

struct WeakModel
{
	virtual ~WeakModel() {}
	virtual void Train(const ClassificationDataset& data_set) = 0; 
	virtual LinearClassifier<RealVector>& GetClassifier() = 0; 
}; 

struct LogisticRegressionModel : public WeakModel
{
	LinearClassifier<RealVector> classifier; 
	LogisticRegression<RealVector> trainer; 
	void Train(const Classificationdataset& data_set) override
	{
		trainer.train(classifier, data_set); 
	}
	
	LinearClassifier<RealVector>& GetClassifier() override {return classifier; }
}; 

struct LDAModel : public WeakModel
{
	LinearClassifier<RealVector> classifier; 
	LDA trainer; 
	void Train(const ClassificationDataset& data_set) override
	{
		trainer.train(classifier, data_set); 
	}

	LinearClassifier<RealVector>& GetClassifier() override {return classifier; }
}; 

const double SVM_C = 100.0; 

struct LinearSVMModel : public WeakModel
{
	LinearClassifier<RealVector> classifier; 
	LinearCSvmTrainer<RealVector> trainer{SVM_C, false}; 
	void Train(const ClassificationDataset& data_set) override
	{
		trainer.train(classifier, data_set); 
	}

	LinearClassifier<RealVector>& GetClassifier() override {return classifier; }
}; 

void StockingEnsemble(const ClassificationDataset& train, const ClassificationDataset& test)
{
	size_t num_patitions = 10; 

	/*weak models*/
	std::vector<std::shared_ptr<WeakModel>> weak_models; 
	weak_models.push_back(std::make_shared<LogisticRegressionModel>()); 
	weak_models.push_back(std::make_shared<LDAModel>()); 
	weak_models.psuh_back(std::make_shared<LinearSVMModel>()); 

	ClassificationDataset train_data_set = train; 
	train_data_set.makeIndependent(); 

	bool removeMean = true; 
	Normalizer<Realvector> normalizer; 
	NormalizeComponentsUnitVariance<RealVector> normalizing_trainer(removeMean); 
	normalizing_trainer.train(normalizer, train_data_set.inputs()); 
	train_data_set = transformInputs(train_data_set, normalizer); 

	PCA pca(train_data_set.inputs()); 
	LinearModel<> pca_encoder; 
	pca.encoder(pca_encoder, 5); 
	train_data_set = transformInputs(train_data_set, pca_encoder); 
	std::cout << "Data normlized, num dimensions " << dataDimension(train_data_set.inputs()) << std::endl; 

	/*train weak models for predictions*/
	for (auto weak_model : weak_models)
	{
		weak_model->Train(train_data_set); 
	}
	
	std::cout << "Weak models trained" << std::endl; 

	/*Generate meta dataset*/
	ClassificationDataset meta_data_train; 
	auto folds = createCVSameSizeBalanced(train_data_set, num_patitions); 
	for (std::size_t i = 0; i != folds.size(); ++i)
	{
		/*access the fold*/
		ClassificationDataset training = folds.training(i); 
		ClassificationDataset validation = folds.validation(i); 

		/*train local weak model - new ones on each folds*/
		std::vector<std::shared_ptr<WeakModel>> local_weak_models; 
		local_weak_models.push_back(std::make_shared<LogisticRegressionModel>()); 
		local_weak_models.push_back(std::make_shared<LDAModel>()); 
		local_weak_models.push_back(std::make_shared<LinearSVMModel>());

		std::vector<Data<unsigned int>> meta_predictions; 
		for (auto weak_model : local_weak_models)
		{
			weak_model->Train(training); 
			auto predictions = weak_model->GetClassifier()(validation.inpus()); 
			meta_predictions.push_back(predictions); 
		}

		/*calcualte meta features*/
		meta_deta_train.append(MakeMetaSet(meta_predictions, validation.labels())); 
	}

	std::cout << "Meta dataset made" << std::endl; 

	/*train meta model*/
	LinearClassifier<RealVector> meta_model; 
	LinearCSvmTrainer<RealVector> trainer(SVM_C, true); 
	trainer.train(meta_model, meta_data_train); 

	std::cout << "Meta algo tained" << std::endl; 

	/*evaluate ensemble*/
	ClassificationDataset test_data_set = test; 
	test_data_set.makeIndependent(); 
	test_data_set = transformInputs(test_data_set, normlizer); 
	test_data_set = transformInputs(test_data_set, pca_encoder); 

	std::cout << "Eval Data normlized, num dimension " << dataDimension(test_data_set.inputs()) << std::endl; 

	std::vector<Data<unsigned int>> meta_predictions; 
	for (auto weak_model : weak_models)
	{
		auto predictions = weak_model->GetClassifier()(test_data_set.inputs()); 
		meta_predictions.push_back(predictins); 
	}

	ClassificationDataset meta_test = MakeMetaSet(meta_predictions, test_data_set.labels()); 

	/*compute errors*/
	ZeroOneLoss<unsigned int> loss; 
	Data<unsigned int> predictions = meta_model(meta_test.inputs()); 
	double accuracy = 1. - loss.eval(meta_test.labels(), predictions); 
	std::cout << "Stacking ensemble accuracy = " << accuracy << std::endl;
}

std::string PrepareDataset(const std::string& dataset_name)
{
	auto new_dataset_name = dataset_name + ".csv"; 
	if (fs::exists(dataset_name))
	{
		std::ifstream file(dataset_name); 
		std::ofstream out_file(new_dataset_name); 
		std::string line; 
		while(std::getline(file, line))
		{
			std::regex re("[\\s,]+"); 
			std::sregex_token_iterator it(line.begin(), line.end(), re, -1); 
			std::sregex_token_iterator reg_end; 
			std::vector<std::string> tokens;
			for (int i = 0; it != reg_end; ++it, ++i)
			{
				if (i == 0)
				{
				continue;	/*skip*/
				} else if (i == 1)
				{
					if (it->str() == "M")
					{
						tokens.push_back("0"); 
					} else
					{
						tokens.push_back("1"); 
					}else
					{
						tokens.push_back(it->str()); 
					}
					
					tokens.push_back(", "); 
				}
				tokens.resize(tokens.size() -1); 
				for (auto& token : tokens)
				{
					out_file << token;
				}
				out_file << "\n"; 
			}
		}
		return new_dataset_name; 
	
}

int main(int argc, char** argv)
{
	if (argc > 1)
	{
		auto dataset_name = fs::path(argv[1]); 
		if (fs::exists(dataset_name))
		{
			dataset_name = PrepareDataset(dataset_name); 
			ClassificationDataset data; 
			importCSV(data, dataset_name, LabelPosition::FIRST_COLUMN); 
			std::size_t num_samples = data.numberOfElements(); 
			std::size_t num_features = dataDimension(data.inputs());
			std::size_t num_classes = numberOfClasses(data.labels()); 

			std::cout << dataset_name << "\n" << "num features: " << num_features << " num classes: " << num_classes << std::endl; 

			/*split data set in the training and testing parts*/
			ClassificationDataset test_data = splitAtElement(data, 500); 

			/*auto classes = classSizes(test_data.labels());*/
			RFClassification(data, test_data); 

			StackingEnsemble(data, test_data); 

			return 0; 
		}
	}

	std::crr << "Dataset file is missed or incorrect \n"; 
	return 0; 
}
