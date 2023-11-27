/*"Dlib" has the "Dlib::vector_normalizer" class. 
 * This method is best for batch training due to the limitation for one big matrix*/
std::vector<matrix<double>> samples; 
...
vector_normalizer<matrix<double>> normalizer; 
samples normalizer.train(samples); 
samples = normalizer(samples); 

/*We see that the object of this class can be reused, but it should be trained at first. like below*/
matrix<double> m(mean(mat(samples))); 
matrix<double> sd(reciprocal(stddev(mat(samples)))); 
for (size_t i = 0; i < samples.size(); ++i)
	samples[i] = pointwise_multiply(samples[i] - m, sd); 

/*Notice that the "Dlib::mat()" function has different overloads fo rmatrix creation from different sources. 
 * We use the "reciprocal()" function that makes the m` = (1/m) matrix if the m is the input matrix.*/

/*Printing matrices for debugging purpose in the "Dlib" lib can be done with the simple stream operator*/
std::cout << mat(samples) << std::endl; 
