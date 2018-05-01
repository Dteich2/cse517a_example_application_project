#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:13:13 2018

@author: gregyork
"""

from sklearn import preprocessing
from sklearn import gaussian_process as GP
from sklearn.model_selection import cross_val_score
import pandas

wineData = pandas.read_csv('/Users/gregyork/Downloads/wine-combined.csv', sep=";")

Y=wineData['quality']

X=wineData.drop(['quality'],axis=1)

le = preprocessing.LabelEncoder()
le.fit(["red","white"])
X['type']=le.transform(X['type'])

rbfobj = GP.kernels.RBF()
ckobj = GP.kernels.ConstantKernel()

rbfReg = GP.GaussianProcessRegressor(kernel=rbfobj)
print("GP rbf Complete")
#rbfClf= GP.GaussianProcessClassifier(kernel=rbfobj)

ckReg = GP.GaussianProcessRegressor(kernel=ckobj)
print("GP dp Complete")
#dpClf = GP.GaussianProcessClassifier(kernel=dpobj)

#reg.fit(X,Y)

rbfRegScore = cross_val_score(rbfReg,X,Y,cv=10,n_jobs=2,scoring='mean_squared_error').mean()
#print("RBF CV Complete")
#rbfClfScore = cross_val_score(rbfClf,X,Y,cv=10,n_jobs=2).mean()

ckRegScore = cross_val_score(ckReg,X,Y,cv=10,n_jobs=2,scoring='mean_squared_error').mean()
#print("DP CV Complete")
#ckClfScore = cross_val_score(dpClf,X,Y,cv=10,n_jobs=2).mean()
