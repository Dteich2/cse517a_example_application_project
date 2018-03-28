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
dpobj = GP.kernels.DotProduct()

rbfReg = GP.GaussianProcessRegressor(kernel=rbfobj)
rbfClf= GP.GaussianProcessClassifier(kernel=rbfobj)

dpReg = GP.GaussianProcessRegressor(kernel=dpobj)
dpClf = GP.GaussianProcessClassifier(kernel=dpobj)

#reg.fit(X,Y)

rbfRegScore = cross_val_score(rbfReg,X,Y,cv=10,n_jobs=2).mean()
rbfClfScore = cross_val_score(rbfClf,X,Y,cv=10,n_jobs=2).mean()

dpRegScore = cross_val_score(dpReg,X,Y,cv=10,n_jobs=2).mean()
dpClfScore = cross_val_score(dpClf,X,Y,cv=10,n_jobs=2).mean()

