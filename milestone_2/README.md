Milestone 2
===========

For this milestone, we used Gaussian Processes applying three different kernels -- RBF, DP (dot product) and constant kernel. We initially tried regression and classification just using RBF and DP for categorizing wine score (scale 1-10). However, this was a bit overzealous as the DP Classification never completed (left running on local computer for over 4 hours). Since RBF finished with the full dataset, we tried smaller datasets to run DP on and got results for 1%, 5%, and 10% of the dataset. We also tried regression using the third kernel, constant kernel, on the data set. This ended up giving us the best overall score.

We ran 10-fold cross validation on all test cases, using (negative) mean-squared error for regressions and accuracy for classifications. 

Results can be found at onepercent, fivepercent and tenpercent for the smaller data set runs. wine.py was used for this evaluation. For RBF and Constant Kernel regression on the full data set, results can be found at fullset. WineGP.py was used for this test.

Difficulties: The DP tests did not finish, as they took too much time, and sometime also returned an error for kernel matrix not being positive definite. We will look into that issue in the future.

Resources: Python and scikit-learn

To run the code: Self explanatory, run regressions / classifications using Python.
