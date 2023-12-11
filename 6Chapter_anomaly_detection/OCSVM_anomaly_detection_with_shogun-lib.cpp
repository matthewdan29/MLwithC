auto csv_file = some<CCSVFile>(dataset_name.string().c_str()); 
Matrix data; 
data.load(csv_file); 

Matrix train = data.submatrix(0, 50); 
train = train.clone(); 
Matrix test = data.submatrix(50, data.num_cols); 
test = test.clone(); 

/*create a dataset*/
auto features = some<CDenseFeatures<DataType>>(train); 
auto test_features = some<CDenseFeatures<DataType>>(test); 

auto gauss_kernel = some<CGaussianKernel>(features, features, 0.5); 

auto c = 0.5; 
auto svm = some<CLibSVMOneClass>(c, gauss_kernel); 
svm->train(features); 

double dist_threshold = -3.15; 

auto detect = [&](Some<CDenseFeatures<DataType>> data)
{
	auto labels = svm->apply(data); 
	for (int i = 0; i < labels->get_num_labels(); ++i)
	{
		auto dist = labels->get_value(i); 
		if (dist > dist_threshold)
		{
			/*Do something with anomalies*/
		} else 
		{
			/*Do something with normal*/
		}
	}
};

detect(features); 
detect(test_features); 


/*
 * 		1) We loaded the dataset from the CSV file so that its an object of the "Matrix" type and split it into two parts for training and testing. 
 *
 * 		2) We declared objects of the "CDenseFeatures" type in order to use loaded data in the Shogun algo
 *
 * 		3) We dclared the kernel object of the "CGaussianKernel" type and used it to initialize the SVM algorithm object of the "CLibSVMOneClass" type. 
 *
 * 		4) Next we had the SVM object in place, we used the "train()" method with the training dataset to fit the algo to our data. 
 *
 * 		5) we defined a distance threshold and used "apply()" method on each of the datasets to detect anomalies. */
