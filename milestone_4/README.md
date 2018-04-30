README
------

For this milestone, we ran a Neural Network on our wine data, and tested it for regression.
In Python, we used MLPRegressor from sklearn.neuralnetwork and fit it to our wine data. We tested out several different options 
for gradient descent, batch size, loss function, and number of nodes and hidden layers. 
We ended up getting the best results using adam (Adaptive Moment Estimation) for gradient descent, which was more efficient and faster 
than sgd (stochastic gradient descent). We generally got better results with smaller batch sizes (10).
We tried both relu (Rectified Linear Unit) and logistic for our loss function, and found that while logistic gave us a smaller CV error,
it took much longer to run.
We used one hidden layer of 20 nodes. We tried larger network structures (e.g. 3 layers of 100, or 5 layers of 20, 30, 40, 30, 20), 
and didn't find any better CV errors.
Difficulties: Often times the regressions took a long time to evaluate when we were testing our networks
Resources: Pandas and scikit-learn
How to run the code: Just run it through Python, if you have the .py file and the csv (Redirect the CSV path to yours)

Our best run: Logistic loss function, adam descent, 1 hidden layer of 20 nodes, batch size of 10, alpha = 0.001
Results: 4.57 s average run time, Cross-validation mean-squared error of 0.54.