README
------

For this milestone, we ran a Neural Network on our wine data, and tested it for regression.
In Python, we used MLPRegressor from sklearn.neuralnetwork and fit it to our wine data. We tested out several different options 
for gradient descent, batch size, loss function, and number of nodes and hidden layers. 
We ended up using adam (Adaptive Moment Estimation) for gradient descent, a batch size of 10, relu (Rectified Linear Unit) for
our loss function, and one hidden layer of 20 nodes (we tried larger network structures, and didn't find any better CV errors)
Difficulties: Often times the regressions took a long time to evaluate when we were testing our networks
Resources: Pandas and scikit-learn
How to run the code: Just run it through Python, if you have the .py file and the csv (Redirect the CSV path to yours)