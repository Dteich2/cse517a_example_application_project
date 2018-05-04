README
------

For our final comparison, we chose to test three models that had shown promising results: Ridge Regression, a decision tree regressor, and a Neural Net.
We ran 10-fold cross-validation 10 times with each model, making sure to split the data identically. In doing so, we gathered 100 negative mean squared error scores for each model.

For each possible pair within the three, we attempted to reject a null hypothesis H0 that their performance was the same using a two-tailed p-test.
We were able to reject all three of these hypotheses within a 95% confidence interval.

If we treat the sign of each t-value as an indicator of performance, we can “rank” our three models as follows:

Ridge Regression > Neural Net > Tree

For this milestone, we also ran a Neural Network on our wine data, and tested it for regression.
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
