from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import pandas

wineData = pandas.read_csv('wine-combined.csv', sep=",")

Y=wineData['quality']

X=wineData.drop(['quality'],axis=1)

le = preprocessing.LabelEncoder()
le.fit(["red","white"])
X['type']=le.transform(X['type'])

reg = linear_model.LogisticRegressionCV(Cs=10, max_iter=1000)

scores = cross_val_score(reg, X, Y, cv=10)
print(scores)