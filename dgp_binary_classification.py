import numpy as np
from scipy.stats import norm
#import pandas
import argparse

from models import ClassificationModel

parser = argparse.ArgumentParser(description='Read in data; perform deep GP binary classification with sghmc inference')
parser.add_argument('data_path')
parser.add_argument('depth', default = 3)
parser.add_argument('output_filename', default = 'DeepGP/results')
args = parser.parse_args()

path = args[0]
depth = args[1]
output_filename = args[2]

# import data 
df = pandas.read_csv(path, sep = ",", header=None)
data = df.values
data = data.astype(int)

# interesting if preprocessing the data first would help. Standardize each feature to be standard normal.
Y = data[:,0]
X = data[:,1:]

# preprocess
X = X - np.mean(X,axis = 0)
sd = np.std(X,axis = 0)
sd[sd == 0] = 1
X = X/sd

# apply the sghmc procedure
model = ClassificationModel(depth)
model.fit(X, Y)

#get posterior samples
m, v, ms = model.predict(X)

ms = ms.reshape((ms.shape[0],ms.shape[1]))
np.savetxt(output_filename, ms)