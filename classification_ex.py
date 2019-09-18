import numpy as np
from scipy.stats import norm
import pandas

from models import RegressionModel, ClassificationModel

from sklearn.metrics import confusion_matrix

# simple dataset (binary classification)
# using Wisconsin cancer dataset
path = './data/breast-cancer-wisconsi-data.csv'
df = pandas.read_csv(path, header=None)
df.replace('?', np.NaN, inplace = True)
df.dropna(axis = 0, inplace = True)


data = df.values[:,1:]
data = data.astype(int)


X_full = data[:, :-1]
Y_full = (data[:, -1:] - 2)/2
Y_full = Y_full.astype(int)


N = X_full.shape[0]
n = int(N * 0.8)
ind = np.arange(N)

np.random.shuffle(ind)
train_ind = ind[:n]
test_ind = ind[n:]

X = X_full[train_ind]
Xs = X_full[test_ind]
Y = Y_full[train_ind]
Ys = Y_full[test_ind]

X_mean = np.mean(X, 0)
X_std = np.std(X, 0)
X = (X - X_mean) / X_std
Xs = (Xs - X_mean) / X_std

model = ClassificationModel()
model.fit(X, Y)

m, v, ms = model.predict(Xs)

preds = np.round(m[:,0],0) 
c = confusion_matrix(preds, Ys)
print("Correctly Classified: {}".format((c[0,0] + c[1,1])/c.sum() ))


# can we run RATE with this?
ms = ms.reshape((ms.shape[0],ms.shape[1]))
np.savetxt('posteriorsamples_wisconsin.txt', ms)
np.savetxt('designmatrix_wisconsin.txt', Xs)

# feed these samples into RATE -- in RATE_deepgp.R

# can we run logistic regression?

# can we run lasso logistic regression?
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

reg_params = np.power(10.,np.array([0,-0.5,-1,-1.5,-2,-2.5,-3]))
coefs = np.zeros((len(reg_params),X.shape[1]))

for i,reg_param in enumerate(reg_params):
	log = LogisticRegression(penalty='l1', solver='liblinear', C = reg_param)
	log.fit(Xs, np.ravel(Ys))
	coefs[i,:] = log.coef_

coefs

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for i in range(X.shape[1]):
	ax.plot(reg_params[::-1], coefs[::-1,i], label = i)

ax.set_xscale('log')
plt.legend()
plt.gca().invert_xaxis()
plt.show()


