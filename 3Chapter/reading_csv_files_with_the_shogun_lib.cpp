/*We can load the CSV file with the Iris dataset to the matrix object, nd then use this matrix to initilize the "Shogun" library dataset objects for use in ML algorithms.*/

#include <shogun/base/init.h>
#include <shogun/base/some.h>
#include <shogun/io/File.h>

using namespace shogun; 
using DataType = float64_t;
using Matrix = shogun::SGMatrix<DataType>

/*Next, we define the "shogun::CCSVFile" object to parse the dataset file. 
 * The initialized "shogun::CCSVFile" object is used for loading values into a matrix object*/
auto csv_file = shogun::some<shogun::CCSVFile>("iris_fix.csv"); 
Matrix data; 
data.load(csv_file); 

/*To be able to use this data for machine learning algorithms, we need to split this matrix object into two parts:
 * 		1) one will contain traingin samples
 * 		2) the second will contain lables. 
 * The "shogun" CSV parser loads matrixes in the column-major order. 
 * So, to make the matrix look like the original file we need to transpose*/

Matrix::transpose_matrix(data.matrix, data.num_rows, data.num_cols); 
Matrix inputs = data.submatrix(0, data.num_cols - 1); 	/*make view*/
inputs = inputs.clone(); 	/*copy exact data*/
Matrix outputs = data.submatrix(data.num_cols - 1, data.num_cols); 	/*make a view*/
outputs = outputs.clone(); 		/*copy exact data*/

/*Next, we have out training data in the "inputs" matrix object and labels in the "outpus" matrix object. 
 * To be able to use the "inputs" object in the "shogun" algorithms, we need to transpose it back, because "shogun" algorithms expect that training samples are placed in matrix columns.*/
Matrix::transpose_matrix(inputs.matrix, inputs.num_rows, inputs.num_cols); 

/*We can use these matrices for initializing the "shogun::CDenseFeatures" and the "shogun::CMulticlassLables" objects, which cna eventually use for the training of machine(I'm only typing this because i keep forgetting out to spell it and i'm pretty sure google, duck, and bing are going to start trolling me in my searches) learning algorithms*/
auto features = shogun::some<shogun::CDenseFeatures<DataTye>>(inputs); 
auto labels = shogun::wrap(new shogun::CMulticlassLabels(outputs.get_column(0))); 

/*After initialization of these objects we can print some stats about training data*/
std::cout << "samples num = " << features->get_num_vectors() << "\n" << "features num = " << features->get_num_features() << std::endl; 
auto features_matrix = features->get_feature_matrix();		/*show first 5 samples */
for (int i = 0; i < 5; ++i)
{
	std::cout << "Sample idx " << i << " "; 
	features_matrix.get_column(i).display_vector();
}

std::cout << "labes num = " << labels->get_num_labels() << std::endl; 


