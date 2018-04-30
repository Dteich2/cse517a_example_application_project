#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:39:27 2018

@author: gregyork
"""

import pandas
#import numpy
#import sklearn
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import gaussian_process as GP
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

#Data load and slicing into X, Y
wineData = pandas.read_csv('/Users/gregyork/Downloads/wine-combined.csv', sep=";")

wineData=wineData.drop(['type'],axis=1)

wineData=wineData.sort_values(by=['quality'])

Y=wineData['quality']

X=wineData.drop(['quality'],axis=1)

qualityColors = {3:"black",4:"purple",5:"blue",6:"green",7:"yellow",8:"orange",9:"red"}
#Plot all eigenvalues to see which ones represent the features with the most info
kpca = KernelPCA(n_components=11, kernel='linear')
eigval = kpca.fit_transform(X)
eigenvals=kpca.lambdas_
plt.bar(range(len(eigenvals)), eigenvals)
plt.show()

#Reduce Data down to 3 dimensions for plotting
kpca = KernelPCA(n_components=3, kernel='linear')
transformedData = kpca.fit_transform(X)
fig2 = plt.figure(0)
ax = fig2.add_subplot(111, projection='3d')
ax.legend(qualityColors)
for q in range(3,10):
    ax.scatter(transformedData[wineData['quality']==q][:,0], transformedData[wineData['quality']==q][:,1], transformedData[wineData['quality']==q][:,2], c=qualityColors[q])
ax.legend(qualityColors)
plt.show()

#Perform relevant Kernal PCA on reduced dimensions to compare results with original run
kpca = KernelPCA(n_components=1, kernel='linear')
transformedData = kpca.fit_transform(X)
print("Gaussian Process")
Y=wineData['quality']
X=transformedData

ckobj = GP.kernels.ConstantKernel()
ckReg = GP.GaussianProcessRegressor(kernel=ckobj)
ckRegScore = cross_val_score(ckReg,X,Y,cv=10,n_jobs=2,scoring='mean_squared_error').mean()

print("Linear Regression")
ridgeReg = linear_model.Ridge(alpha=0.01)
ridgeRegScore = cross_val_score(ridgeReg,X,Y,cv=10,n_jobs=2,scoring='mean_squared_error').mean()
print("GP Constant Kernel Regression CV Score:",ckRegScore)
print("Linear Regression CV Score:",ridgeRegScore)


