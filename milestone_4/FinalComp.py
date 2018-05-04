from sklearn import linear_model
from sklearn import tree
from sklearn.gaussian_process import kernels
from sklearn.neural_network import MLPRegressor
from sklearn import gaussian_process
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold

import matplotlib.pyplot as plt

import pandas
import numpy

wineData = pandas.read_csv('wine-combined.csv', sep=",")

Y=wineData['quality']

X=wineData.drop(['quality'],axis=1)

le = preprocessing.LabelEncoder()
le.fit(["red","white"])
X['type']=le.transform(X['type'])

ridgeReg = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
clf = tree.DecisionTreeRegressor()
reg = linear_model.LogisticRegressionCV(Cs=10, max_iter=100)
NN = MLPRegressor((20,),'relu','adam',batch_size=10,learning_rate='constant',alpha=.001,max_iter=1000000,random_state=3,tol=1e-8)

NUM_RUNS = 10
NUM_FOLDS = 10

#leaving out logistic for now because it's quiiiite slow
models = [ridgeReg, clf, NN]

scores = {model : [] for model in models}
crosses = []

for i in range(NUM_RUNS):
	crosses.append(KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=i))

for model in models:
	for cross in crosses:
		scores[model] += cross_val_score(model,X,Y,cv=cross,n_jobs=2,scoring='neg_mean_squared_error').tolist()

means = {model : numpy.mean(scores[model]) for model in models}
pairs = [(ridgeReg,clf),(ridgeReg,NN),(clf,NN)]

for A,B in pairs:
	dbar = means[A] - means[B]
	sigmad = (sum((scores[A][i] - scores[B][i] - dbar)**2 for i in range(NUM_FOLDS * NUM_RUNS)) / ((NUM_FOLDS * NUM_RUNS) - 1))**0.5
	tval = dbar / (sigmad / ((NUM_FOLDS * NUM_RUNS)**0.5))

	print("THIS IS ONE")
	print(dbar)
	print(sigmad)
	print(tval)

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

axs[0].hist(scores[ridgeReg], bins=50)
axs[1].hist(scores[clf], bins=50)
axs[2].hist(scores[NN], bins=50)
#axs[3].hist(scores[reg], bins=50)

plt.show()