/*"shogun" has the "CLinearRidgeRegression" class for solving simple linear regression problems. */
auto x = some<CDenseFeatures<float64_t>>(x_values); 
auto y = some<CRegressionLables>(y_values); 		/*real-valued lables*/
float64_t tau_regularization = 0.001; 
auto lr = some<CLinearRidgeRegression>(tau_regularization, nullptr, nullptr); /*regression model with regularization*/
lr->set_labels(y); 
r->train(x); 

/*For new x inputs, we can predict new y values */
auto new_x = some<CDenseFeatures<float64_t>>(new_x_values); 
auto y_predict = lr->apply_regreesion(new_x);

/*We can get the calculated parameters (the linear regression task solution) vector*/
auto weights = lr->get_w(); 

/*We can now calculate the value of MSE */
auto y_predict = lr->apply_regression(x); 
auto eval = some<CMeanSquaredError>(); 
auto mse = eval->evaluate(y_predict, y); 


