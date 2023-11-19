/*"LinearRegression" class, which provides analytical solutions. and the "LinearSAGTrainer" class which provides a stochastic average gradient iterative method,*/

using namespace shark; 
using namespace std; 
Data<RealVector> x; 
Data<RealVector> y; 
RegressionDataset data(x,y); 
LinearModel<> model; 

LinearRegression trainer; 
trainer.train(model, data); 

/*We can get the calculated parameters vecotr*/
auto b = model.parameterVector(); 

/*For new x inputs we can predict the new y values */
Data<RealVector> new_x; 
Data<RealVector> prediciton = model(new_x); 

/*We can calculate the value of squared error*/
SquaredLoss<> loss; 
auto se = loss(y, prediction);

