Milestone 1

We got our wine data from UCI (details in the main database README), combined the red and white databases into one, and then used Python scikit learn to run a linear regression algorithm on the data with 10-fold cross validation.
We also performed a CART decision tree algorithm on the data (also with 10-fold CV)

Difficulties: Deciding whether the output variable (the score, 1-10 that the wine received) should be treated as categorical or linear. We tried both, and for now we think we are using it as a linear variable.

To run the code, make sure that the wine dataset is in the same folder as your code, and run the Python script.
