/*linear discriminant analysis algorithm which can be used for dimensionality reduction. 
 * Its a supervised algorithm, so it needs labaled data. 
 * This algorithm is implmemnted with the "Dlib::compute_lda_transform()" function, which takes 4 parameters. 
 * 		
 * 		1) is the input/output parameter as input, it is used to pass inp training data and as output, it receives the LDA transformation matrix. 
 *
 * 		2) second is the output for the mean values
 *
 * 		3) third is the labels for the input data, 
 *
 * 		4) the fourth is the desired number of target timensions. */
void LDAReduction(const Matrix &data, const std::vector<unsigned long> &labels, unsigned long target_dim)
{
	Dlib::matrix<DataType, 0, 1> mean; 
	Matrix transform = data; 
	Dlib::compute_lda_transform(transform, mean, labels, target_dim); 

	for (long r = 0; r < data.nr(); ++r)
	{
		Matrix row = transfrom * Dlib::trans(Dlib::rowm(data, r)) - mean; 
		double x = row(0, 0); 
		double y = row(1, 0); 
	}
}

/*To perform an actual LDA transform after the algorthm has been trained, we multiply our samples with the LDA matrix. 
 * In our case, we also transposed them. */
tansform * Dlib::trans(Dlib::rowm(data, r)); 
