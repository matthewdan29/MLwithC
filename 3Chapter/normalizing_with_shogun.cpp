/*The "shogun::CRescaleFeatures" class in the "shogun" library implements min-max normalization. 
 * We can reuse objects of this class for scaling different data with the same learned statistics. 
 * It can be useful in cases when we train a machine learning algorithm on one data format with applied rescaling, and then we use the algorithm for predictions on new data. 
 * To make this new data work as we want, we have to rescale new data in the same way as we did in the training process.*/
#include <shogun/preprocessor/RescaleFeatures.h>
...
auto features = shogun::some<shogun::CDenseFeatures<DataType>>(inputs); 
...
auto scaler = shogun::wrap(new shogun::CRescaleFeatures()); 
scaler->fit(features); 			/*Learn statistics - min and max values*/
scaler->transform(features); 		/*apply scaling*/


/*To learn statistics values, we use the "fit()" method and for features modification, we use the "transform()" method of the "CRescaleFeatures" class*/

/*We can print updated features with the "display_vector()" method of the "SGVector" class.*/
auto features_matrix = features->get_feature_matrix(); 
for (int i = 0; i < n; ++i)
{
	std::cout << "Simple idx" << i << " ";
	features_matrix.get_column(i).display_vector(); 
}

/*Read the "shogun" lib Doc's due to the fact some algorithms perform normalization of input data as an internal step(Think of it like having the rand() in the first part of quick sort to make sure that the partision is not going in the worse case, then look and see how other algo's build on top of n-quicksort() like some DFS or BFS algo's)*/
