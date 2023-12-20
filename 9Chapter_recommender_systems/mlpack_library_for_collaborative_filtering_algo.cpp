/*This define the sparse matrix for ratings.*/
arma::SpMat<DataType> ratings_matrix(ratings.size(), movies.size()); 
std::vector<std::string> movie_titles; 
{
	/*fill matrix*/
	movie_titles.resize(movies.size()); 

	size_t user_idx = 0; 
	for (auto& r : ratings)
	{
		for (auto& m : r.second)
		{
			auto mi = movies.find(m.first); 
			auto movie_idx = std::distance(movies.begin(), mi); 
			movie_titles[static_cast<size_t>(movie_idx)] = mi->second; 
			ratings_matrix(user_idx, movie_idx) = static_cast<DataType>(m.second); 
		}
		++user_idx; 
	}
}

/*The "mlpack::cf::CFType" class object takes the parameters in the constructor: 
 * 		1) The rating matrix, 
 * 		2) The matrix decomposition policy
 * 		3) The number of neighbors
 * 		4) The number of target factors
 * 		5) the number of iterations
 * 		6) the minimum value of learning error*/

distance(X.col(i), X.col(j)) = distance(w H.col(i), w H.col(j)); 

/*This expression can be seen as the nearest neighbor search on the 'H' matrix with the Mahalanobis distance*/
/*factorization rank*/
size_t n_factors = 100; 
size_t neighborhood = 50; 

mlpack::cf::NMFPolicy decomposition_policy; 

/*stopping criterions*/
size_t max_iterations = 20; 
double min_residue = le-3; 

mlpack::cf::CFType cf(ratings_matrix, decomposition_policy, neighborhood, n_factors, max_iterations, min_residue); 

/*Recommendations can be retrieved with the "GetRecommendations" method. 
 * This method gets the number of reommendations you want to get, the output matrix for recommendations and the list of user IDs for users you want to get recommendations from*/
arma::Mat<size_t> recommendations; 
/*Get 5 recommendations for specified users.*/
arma::Col<size_t> users; 
users << 1 << 2 << 3; 

cf.GetRecommendations(5, recommendations, users); 

for (size_t u = 0; u < recommendations.n_cols; ++u)
{
	std::cout << "User " << users(u) << " recommendations are: "; 
	for (size_t i = 0; i < recommendations.n_rows; ++i)
	{
		std::cout << movie_titles[recommendations(i, u)] << ";"; 
	}
	std::cout << std::endl;
}
