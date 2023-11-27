/*There are no functions for data normalization in the "Eigen" lib. 
 * We can implement them according to the provided formulas. */

/*For the standardization, we first have to calculate the standard deviation.*/
Eigen::Array<double, 1, Eigen::Dynamic> std_dev = ((X.rowwise() - X.colwise().mean()).array().square().colwise().sum() / (x_data.rows() -1)).sqrt(); 

/*Having the standard deviation value, the rest of the formula for standardization will look like below code*/
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_std = (x.rowwise() - x.colwise().mean()).array().rowwise() / std_dev; 

/*Implementation of "min_max" normalization is very straightforward and does not require intermediate values, */
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_min_max = (x.rowwise() - x.colwise().minCoeff()).array().rowwise() / (x.colwise().maxCoeff() - x.colwise().minCoeff()).array(); 

/*We implement the "mean" normalization in the same way*/
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x_avg = (x.rowwise() - x.colwise().mean()).array().rowwise() / (x.colwise().maxCoeff() - x.colwise().minCoeff()).array(); 

/*Notice that the implementation of the forumlas was vectorized way without loops; this approach is mor computationally effcient because it can be compiled for execution on a GPU or a CPU single instruction Multiple Data (SIMD) instructions*/
