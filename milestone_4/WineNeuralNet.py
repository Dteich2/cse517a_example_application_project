#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 12:06:48 2018

@author: gregyork
"""

import pandas
#from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import time

wineData = pandas.read_csv('/Users/gregyork/Downloads/wine-combined.csv', sep=";")

Y=wineData['quality']

X=wineData.drop(['quality'],axis=1)

le = preprocessing.LabelEncoder()
le.fit(["red","white"])
X['type']=le.transform(X['type'])
'''
print("#####CLASSIFIER#####")
#alpha = regularizer term
clf = MLPClassifier((10,10),'relu','sgd',batch_size=200,learning_rate='constant',alpha=.001,max_iter=1000000,random_state=1,tol=1e-8)
start = time.time()
clf.fit(X,Y)
totalRun = time.time() - start

print("Run Stats")
print(totalRun,"Run time")
print(clf.n_iter_,"Iterations")
print(clf.loss_,"Final Loss")

NNscore = -cross_val_score(clf,X,Y,cv=10,n_jobs=2,scoring='neg_mean_squared_error').mean()
print(NNscore, "CV MSE")
'''
totalRunTime=0
totalIters=0
bestCVScore=1000000
print("#####REGRESSOR#####")
for i in range(1,6):
    #alpha = regularizer term
    reg = MLPRegressor((20),'logistic','adam',batch_size=10,alpha=.001,max_iter=10000,random_state=i,tol=1e-8)
    start = time.time()
    reg.fit(X,Y)
    totalRun = time.time() - start

    totalRunTime += totalRun
    totalIters += reg.n_iter_

    print("Run",i,"Stats")
    print(totalRun,"Run time (s)")
    print(reg.n_iter_,"Iterations")
    print(reg.loss_,"Final Loss")

    NNscore = -cross_val_score(reg,X,Y,cv=10,n_jobs=2,scoring='neg_mean_squared_error').mean()
    print(NNscore, "CV MSE")
    bestCVScore=min(bestCVScore,NNscore)
print("\n")
print(totalRunTime/5,"Average Run Time")
print(totalIters/5,"Average # of Iterations")
print(bestCVScore, "Best Cross-Val MSE")