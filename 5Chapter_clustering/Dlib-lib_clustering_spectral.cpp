/*"spectral_cluster" function takes the distance function object, the training dataset, and the number of cluster as parameters. 
 * As a reuslt, it returns a container with cluster indices, which have the same ordering as the input data. 
 * These objects are determined with KNN algorithm, which the Euclidean distance for the distance measure*/
typedef matrix<double, 2, 1> sample_type; 
typedef knn_kernel<sample_type> kernel_type; 
...
std::vector<sample_type> samples; 
...
std::vector<unsigned long> clusters = spectral_cluster(kernel_type(samples, 15), samples, num_clusters); 

