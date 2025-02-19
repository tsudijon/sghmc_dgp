import numpy as np
from scipy.stats import norm
import pandas

from models import ClassificationModel
from sklearn.metrics import confusion_matrix

# import data 
#path = './data/veg_bug_molar_data.csv'
path = './data/PTEN_tumor_data.csv'
df = pandas.read_csv(path, sep = ",", header=None)
data = df.values
data = data.astype(int)

# interesting if preprocessing the data first would help. Standardize each feature to be standard normal.
Y_full  = data[:, 0] # should be /pm 1
Y_full = Y_full.astype(int)
X_full = data[:,1:]

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

# preprocess
X = X - np.mean(X,axis = 0)
sd = np.std(X,axis = 0)
sd[sd == 0] = 1
X = X/sd

# apply the sghmc procedure
model = ClassificationModel(3)
model.fit(X, Y)

#get posterior samples
m, v, ms = model.predict(X)

preds = np.round(m[:,0],0) 
c = confusion_matrix(preds, Y)
print("Correctly Classified: {}".format((c[0,0] + c[1,1])/c.sum() ))

# seems completely overfitted: training error = 1?

ms = ms.reshape((ms.shape[0],ms.shape[1]))
np.savetxt('posteriorsamples_PTEN.txt', ms)
np.savetxt('designmatrix_PTEN.txt', X)

# how to fit models for feature selection?


# early stopping might be something of interest...