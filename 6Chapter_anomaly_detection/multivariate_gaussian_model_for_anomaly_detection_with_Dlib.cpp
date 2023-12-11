/*We can implement anomaly dtection with the multivariate gaussion distribution approach. 
 * Below shows how to implement this approach with the "Dlib" linear algebra routines*/
void multivariateGaussianDist(const Matrix& normal, const Matrix& test)
{
	/*assume that rows are samples and columns are features*/

	/*calculate per feature mean*/
	dlib::matrix<double> mu(1, normal.nc()); 
	dlib::set_all_elements(mu, 0); 

	for (long c = 0; c < normal.nc(); ++c)
	{
		auto col_mean = dlib::mean(dlib::colm(normal, c)); 
		dlib::set_colm(mu, c) = col_mean; 
	}

	/*calculate covariance matrix*/
	dlib::matrix<double> cov(normal.nc(), normal.nc()); 
	dlib::set_all_elements(cov, 0); 
	for (long r = 0; r < normal.nr(); ++r)
	{
		auto row = dlib::rowm(normal, r); 
		cov += dlib::trans(row - mu) * (row - mu); 
	}
	cov *= 1.0 / normal.nr(); 
	double cov_det = dlib::det(cov); 	/*matrix determinant*/
	dlib::matrix<double> cov_inv = dlib::inv(cov); 	/*inverse matrix*/

	/*define probability function*/
	auto first_part = 1. / std::pow(2. * M_PI, normal.nc() / 2.) / std::sqrt(cov_det); 

	auto prob = [&](const dlib::matrix<double>& sample)
	{
		dlib::matrix<double> s = sample - mu; 
		dlib::matrix<double> exp_val_m = s * (cov_inv * dlib::trans(s)); 
		double exp_val = -0.5 * exp_val_m(0, 0); 
		double p = first_part * std::exp(exp_val); 
		return p; 
	}; 

	/*change this parameter to see the decision boundary*/
	double prob_threshold = 0.001; 

	auto detect = [&](auto samples)
	{
		for (long r = 0; r < samples.nr(); ++r)
		{
			auto row = dlib::rowm(samples, r); 
			auto p = prob(row); 
			if (p >= prob_threshold)
			{
				/*Do something with anoalies*/
			} else 
			{
				/*Do something with normal*/
			}
		}
	}; 

	detect(normal); 
	detect(test); 
}

/*The Idea of this approach is to define a function that returns the probability of appearing, given a sample in a dataset. 
 * We calculate the statistcal characteristics of the training dataset. 
 * 	1) We calculate the mean values of each feature and store them into the one dimensional matrix. 
 *
 * 	2) We calculate the covariance matrix for the training samples using the formula for the correlation matrix that was given in the prior theoretical section named denstiy estimation approach for anomaly detection. 
 *
 * 	3) we determine the correlation matrix determinant and inverse version. 
 * 	*/
