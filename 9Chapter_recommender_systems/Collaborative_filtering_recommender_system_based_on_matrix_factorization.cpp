/*1) first we make base type definitions*/
using DataType = float; 
/*Using Eigen::ColMajor is Eigen restriction - todense method always returns*/
/*Matrixes in COlMajor order*/
using Matrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; 

using SparseMatrix = Eigen::SparseMatrix<DataType, Eigen::ColMajor>; 

using DiagonalMatrix = Eigen::DiagonalMatrix<DataType, Eigen::Dynamic, Eigen::Dynamic>; 
/*These definitions allow us to write less source code for matrices' types and to quickly change floating-point precision.*/

/*Next, we define and initialze the ratings matrix, list of movies titles, and binary rating flags matrix.*/
SparseMatrix ratings_matrix; 			/*user-item ratings*/
SparseMatrix p; 				/*binary variables*/
std::vector<std::string> movie_titles; 

/*We have a particular helper function, "LoadMoves", which loads files to the map container.*/
auto movies_file = root_path / "movies.csv"; 
auto movies = LoadMovies(movies_file); 

auto ratings_file = root_path / "ratings.csv"; 
auto ratings = LoadRatings(ratings_file); 

/*Next, After data is loaded, we initialize mattrix objects with the right size*/
ratings_matrix.resize(static_cast<Eigen::Index>(ratings.size()), static_cast<Eigen::Index>(movies.size())); 
ratings_matrix.setZero(); 
p.resize(ratings_matrix.rows(), ratings_matrix.cols()); 
p.setZero(); 
movie_titles.resize(movies.size()); 
/*However, because we've loaded data to the map, we need to move the required rating values to the matrix*/

/*Next, we initialze the movie titles list, convert user IDs to our zero based sequential order, and initialize the binary rating matrix*/
Eigen::Index user_idx = 0; 
for (auto& r : ratings)
{
	for (auto& m : r.second)
	{
		auto mi = movies.find(m.first); 
		Eigen::Index movie_idx = std::distance(movies.begin(), mi); 
		movie_titles[static_cast<size_t>(movie_idx)] = mi->second; 
		ratings_matrix.insert(user_idx, movie_idx) = static_cast<DataType>(m.second); 
		p.insert(user_idx, movie_idx) = 1.0; 
	}
	++user_idx; 
}
ratings_matrix.makeCompressed(); 

/*Next, after the rating matrix is initialized, we define and intialize our training variables*/
auto m = ratings_matrix.rows(); 
auto n = ratings_matrix.cols(); 

Eigen::Index n_factors = 100; 
auto y = InitializeMatrix(n, n_factors); 	/*this matrix corresponds to the user preferences*/
auto x = InitializeMatrix(m, n_factors); 	/*This matrix corresponds to the item parameters*/

/*Next, we defined the number of factors we were interested in after decomposition. 
 * These matrices are initialized with random values and normalized. 
 * Such an approach is used to speed up algorithm convergence.*/
Matrix InitializeMatrix(Eigen::Index rows, Eigen::Index cols)
{
	Matrix mat = Matrix::Random(rows, cols).array().abs(); 
	auto row_sums = mat.rowwise().sum(); 
	mat.array().colwise() /= row_sums.array(); 
	return mat; 
}

/*Next, we define and initialize the regularization matrix and identiy matrices, which are constant during all learning cycles*/
DataType reg_lambda = 0.1f; 
SparseMatrix reg = (reg_lambda * Matrix::Identity(n_factors, n_factors)).sparseView(); 

/*Define diagonal identity terms*/
SparseMatrix user_diag = -1 * Matrix::Identity(n, n).sparseView(); 
SparseMatrix item_diag = -1 * Matrix::Identity(m, m).sparseView(); 

/*Next, because we implement an algorithm version that can deal with implicit data, we need to convert our rating matrix to another view to decrease computational complexity. 
 * Our version of the algorithm needs user ratings in a algo form and as diagonal matrices for every user and item so that we can make two containers with corresponding matrix objects. */
std::vector<DiagonalMatrix> user_weights(static_cast<size_t>(m)); 
std::vector<DiagonalMatrix> item_weights(static_cast<size_t>(n)); 
{
	Matrix weights(ratings_matrix); 
	weights.array() *= alpha; 
	weights.array() += 1; 

	for (Eigen::Index i = 0; i < m; ++i)
	{
		user_weight[static_cast<size_t>(i)] = weights.row(i)asDiagonal(); 
	}
	for (Eigen::Index i = 0; i < n; ++i)
	{
		item_weights[static_cast<size_t>(i)] = weights.col(i).asDiagonal(); 
	}
}

/*Let's define the main learning cycle, which runs for a specified number of iterations.*/
size_t n_iterations = 5; 
for (size_t k = 0; k < n_iterations; ++k)
{
	auto yt = y.transose(); 
	auto yty = yt * y; 
	...; 
	/*update item parameters*/
	...;
	auto xt = x.transpose();
	auto xtx = xt * x; 
	...; 
	/*update users preferences*/
	...; 
	auto w_mse = CalculateWeightedMse(x, y, p, ratings_matrix, alpha); 
}

/*Below shows how to update item parameters*/
#pragma omp parallel
{
	Matrix diff; 
	Matrix ytcuy; 
	Matrix a, b, update_y; 
	#pragma omp for private(diff, ytcuy, a, b, update_y)
	for (size_t i = 0; i < static_cast<size_t>(m); ++i)
	{
		diff = user_diag; 
		diff += user_weights[i]; 
		ytcuy = yty + yt * diff * y; 
		auto p_val = p.row(static_cast<Eigen::Index>(i)).transpose(); 

		a = ytcuy + reg; 
		b = yt * user_weights[i] * p_val; 

		update_y = a.colPivHouseholderQr().solve(b); 
		x.row(static_cast<Eigen::Index>(i)) = update_y.transpose(); 
	}
}

/*Below show how to update users' preferences:*/
#pragma omp parallel
{
	Matrix diff; 
	Matrix xtcux; 
	Matrix a, b, update_x; 
	#pragma omp for private(diff, xtcux, a, b, update_x)
	for (size_t i = 0; i < static_cast<size_t>(n); ++i)
	{
		diff = item_diag; 
		diff += item_weights[i]; 
		xtcux = xtx + xt * diff * x; 
		auto p_val = p.col(static_cast<Eigen::Index>(i)); 

		a = xtcux + reg; 
		b = xt * item_weights[i] * p_val; 

		update_x = a.colPivHouseholderQr().solve(b); 
		y.row(static_cast<Eigen::Index>(i)) = update_x.transpose(); 
	}
}
/*We have two parts of the loop body that are pretty much the same
 * 		1) we updated item parameters with frizzed user options
 * 		2) we updated user preferences with fizzed item parameters. 
 * Notice that all matrix objects were moved outside of the interal loop body to reduce memory allocation and significantly improve program performance. 
 * Also notice that we parallelized the user and item paramters' calculations separately because one of them should always frizzed during the calculation of the oter one. */

/*To estimate the progress of the learning process of our system, we can calculate the Mean Squared Error (MSE) between the original rating matrix and a predicted one. 
 * To calculate the predicted rating matrix we define "RatingsPredictions()"*/
Matrix RatingsPredictions(const Matrix& x, const Matrix& y)
{
	return x * y.transpose(); 
}

/*To calculate the MSE, we can use the expression from our optimization function*/
DataType CalculateWeightedMse(const Matrix& x, const Matrix& y, const SparseMatrix& p, const SparseMatrix& ratings_matrix, DataType alpha)
{
	Matrix c(ratings_matrix); 
	c.array() *= alpha; 
	c.array() += 1.0; 

	Matrix diff(p - RatingsPredictions(x, y)); 
	diff = diff.array().pow(2.f); 

	Matrix weighted_diff = c.array() * diff.array(); 
	return weighted_diff.array().mean(); 
}

/*Please note that we have to use weights and binary ratins to get a meaningful value for the error because a similar approach was used during the learning proces. 
 * It is essential to understand that this algorithm doesn't learn the original scale of ratings, but instead it learns prediction values in the range from 0 to 1. */

/*The following function shows user preferences and system recommendations. 
 * To identify what a user likes, we show movie titles that the user has rated with a rating value of more than 3. 
 * We show movies that the system rates as equal to or higher than a 0.8 rating coefficient to identify which movie the system revommends to the user*/
void PrintRecommendations(const Matrix& ratings_matrix, const Matrix& ratings_matrix_pred, const std::vector<std::string>& movie_titles)
{
	auto n = ratings_matrix.cols(); 
	std::vector<std::string> liked; 
	std::vector<std::string> recommended; 
	for (Eigen::Index u = 0; u < 5; ++u)
	{
		for (Eigen::Index i = 0; i < n; ++i)
		{
			DataType orig_value = ratings_matrix(u, i); 
			if (orig_value >= 3.f)
			{
				liked.push_back(movie_titles[static_cast<size_t>(i)]); 
			}
			DataType pred_value = ratings_matrix_pred(u, i); 
			if (pred_value >= 0.8f && orig_value < 1.f)
			{
				recommended.push_back(movie_titles[static_cast<size_t>(i)]); 
			}
		}
		std::cout << "\nUser " << u << " liked :"; 
		for (auto& l : liked)
		{
			std::cout << l << "; "; 
		}
		std::cout << "\nUser " << u << " recommended : "; 
		for (auto& r : recommended)
		{
			std::cout << r << "; "; 
		}
		std::cout << std::endl; 
		liked.clear(); 
		recommended.clear(); 
	}
}

/*this function can be used as follows: */
PrintRecommendations(ratings_matrix, RatingsPredictions(x, y), movie_titles); 

