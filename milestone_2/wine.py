from sklearn.gaussian_process import kernels
from sklearn import gaussian_process
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import pandas

wineData = pandas.read_csv('wine-combined.csv', sep=",")

Y=wineData['quality']

X=wineData.drop(['quality'],axis=1)

le = preprocessing.LabelEncoder()
le.fit(["red","white"])
X['type']=le.transform(X['type'])

rbf = kernels.RBF()
dotp = kernels.DotProduct()

gp_rbf = gaussian_process.GaussianProcessClassifier(kernel=rbf)
gp_dotp = gaussian_process.GaussianProcessClassifier(kernel=dotp)

print('Training with rbf...')
scores = cross_val_score(gp_rbf, X, Y, cv=10)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print('Training with dot product...')
scores = cross_val_score(gp_dotp, X, Y, cv=10)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))