/*The "Shark-ML" library has a unified API for serializing models of all kinds.
 * Every model has the "write" and "read" methods for saving and loading model parameters, repectively. 
 * These methods take an instance of the "boost::archive" object as an input parameter.*/

/*First, we generate training data for the linear regression model:*/
/*Below, we created two vectors, "x_data" and "y_data", which contain predictor and target value objects of the "RealVector" type 
 * Then, we made the 'x' and 'y' objects of the "Data" type and placed them inot the "data" object of the "RegressopmDataset" type.*/
std::vector<RealVector> x_data(n); 
std::vector<RealVector> y_data(n); 

std::random_device rd; 
std::mt19937 re(rd()); 
std::uniform_real_distribution<double> dist(-1.5, 1.5); 

RealVector x_v(1); 
RealVector y_v(1); 
for (size_t i = 0; i < n; ++i)
{
	x_v(0) = i; 
	x_data[i] = x_v; 

	y_v(0) = func(i) + dist(re); 		/*add noise*/
	y_data[i] = y_v; 
}

Data<RealVector> x = createDataFromRange(x_data); 
Data<RealVector> y = createDataFromRange(y_data); 
RegressionDataset data(x, y); 

/*Below shows how to train a linear model object with the dataset object we initialized:*/
LinearModel<> model; 
LinearRegression trainer; 
trainer.train(model, data); 

/*Now that we've trained the model, we can save its parameters in a file using the "boost::archive::polymorphic_binary_oarchive" object.*/
std::ofstream ofs("shark-linear.dat"); 
boost::archive::polymorphic_binary_oarchive oa(ofs); 
model.write(oa); 

/*Below shows how to load saved models parameters:*/
std::ifstream ifs("shark-linear.dat"); 
boost::archive::polymorphic_binary_iarchive ia(ifs); 
LinearModel<> model; 
model.read(ia); 

/*Instead of using binary serialization, the "shark-ml" library allows us to use the "boost::archive::polymorphic_text_oarchive" and "boost::archive::polymorphic_text_airchive" types to serialize to an ASCII text file.*/
/*below, shows how to generate new test values so that we can check the model:*/
std::vector<RealVector> new_x_data; 
for (size_t i = 0; i < 5; ++i)
{
	new_x_data.push_back({static_cast<double>(i)}); 
	std::cout << func(i) << std::endl; 
}

/*Below code shows how to use the model for prdiction purposes:*/
auto prediction = model(createDataFromRange(new_x_data)); 
std::cout << "Predictions: \n" << prediction << std::endl; 


