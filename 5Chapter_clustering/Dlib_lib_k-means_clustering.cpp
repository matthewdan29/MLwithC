/*As an initial step, we define the required types*/
typedef matrix<double, 2, 1> sample_type; 
typedef radial_basis_kernel<sample_type> kernel_type; 

/*Then, we initialize an object of the "kkmeans" type. 
 * Its constructor takes an object that will define cluster centroids as input parameters. 
 * We can use an object of the "kcentroid" type for this. 
 * Its constructor takes three parameters: 
 * 		1) The object that defines the kernel
 * 		2) The numerical accuracy for the centroid estimation
 * 		3) is the upper limit on the runtime complexity*/
kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8); 
kkmeans<kernel_type> kmeans(kc); 

/*Next step, we initialize cluster center with the "pick_initial_centers()" function. 
 * This function takes the number of clusters the output container for center objects, the taining data and the distance function object as parameters*/
std::vector<sample_type> samples; 
...
size_t num_clusters = 2; 
std::vector<sample_type> initial_centers; 
pick_initial_centers(num_clusters, initial_centers, samples, kmeans.get_kernel()); 

/*When initial centers are selected we can use them for the "kemans::train()" method to determine exact clusters*/
kmeans.set_number_of_centers(num_clusters); 
kmeans.train(samples, initial_centers); 

for (size_t i = 0; i != samples.size(); i++)
{
	auto cluster_idx = kmeans(samples[i]); 
	...
}

