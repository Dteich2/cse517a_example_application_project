# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import cross_val_score
import pandas

wineData = pandas.read_csv('wine-combined.csv', sep=";")

Y=wineData['quality']

X=wineData.drop(['quality'],axis=1)

le = preprocessing.LabelEncoder()
le.fit(["red","white"])
X['type']=le.transform(X['type'])
#reg = linear_model.RidgeCV([0.01, 0.1, 1.0, 10.0, 100.0])
reg = linear_model.Ridge(alpha=0.01)
reg.fit(X, Y)

 scores = cross_val_score(reg, X, Y, cv=10)
 
 #Decision Tree
 from sklearn import tree
 #import graphviz
 
 clf = tree.DecisionTreeClassifier()
 clf = clf.fit(X,Y)
 treeScores = cross_val_score(clf, X, Y, cv=10)
 
 #dot_data = tree.export_graphviz(clf, out_file=None)
 #graph = graphviz.Source(dot_data)
