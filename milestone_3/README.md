Milestone 3
===========

For Dimensionality Reduction, we applied PCA on the 11 features of our wine data with both Linear PCA and RBF Kernel PCA. We did have a 12th categorical feature of which we decided to drop from the data in order to be able to perform PCA. Initially, we applied linear PCA to plot the features corresponding to the 3 highest eigenvalues. We were hoping to see some sort of clustering corresponding to the wine qualities in the data. Unfortunately, the data overlapped in the same cloud in the 3-D plot. This initial reason was the motivation for why we went to RBF. When RBF Kernel PCA was applied to reduce the data to 3-dimensions, there still wasn't seperated clustering of the wine qualities, but a pattern was somewhat noticable in the plot. Plotting the RBF Kernel PCA eigenvalues (in the bugged form) showed that about six features held the majority of the feature information.

However, during demoing, it was pointed out to us that no clustering shouldn't necessarily rule out linear PCA. Additionally, we had a bug in our eigenvalue plotting script. After we fixed the eigenvalue plotting issue at the demo, we retried Linear PCA, and the results showed that just about all of our feature information was held in one feature. Thus we retried Linear PCA retaining only one feature in the PCA transformation.

We ran 2 previous milestone regression models on the transformed datasets: Gaussian Processes Regression and Ridge Regression. Linear PCA was applied to 1-feature, 3-feature (because we had to plot it anyway), and 6-feature dimensionality reduction on the data. We also retained results for 1-feature, 3-feature and 6-feature RBF Kernel PCA for comparison. All of these results can be seen in Results.txt.

All of these results ended up being slightly worse than the regression models we performed in the previous milestones. Though we noted that Gaussian Processes ran about 50% faster (~30 minutes down from ~1 hour). We also noted for linear PCA there was hardly any difference in the 1-feature, 3-feature, and 6-feature run. We surmise this is because the top eigenvalue held just about all of the information of all the feature dimensions.

Difficulties: An initial bug in our eigenvalue plots lead us astray until we fixed it. Also, no obvious clustering of the data when reduced to 3 dimensions with either Linear or RBF Kernel PCA.

Resources: KernelPCA in scikit-learn (python)

To run the code: Execute WineDimRed.py as a python script. This script will, by default, run Linear PCA on all features to obtain and plot the eigenvalues. It will then run PCA again for 3-feature dimension reduction to plot the data in 3-D. Then it will run 1-feature dimension reduction for Linear PCA. Finally, it will run Constant Kernal GP and Ridge Regression to display cross-validation scores for the 1-dimension reduction. This last step may take awhile.

To run RBF rather than linear, the 'linear' should be changed to 'rbf' in the KernelPCA learner. To run on more than 1 feature, the n_components value should be changed to the number of retained features desired.
