import numpy as np
from scipy.stats import norm
import pandas

from models import ClassificationModel


import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def reshape_img(a):
	flat = a.reshape((a.shape[0],a.shape[1]*a.shape[2]))
	return flat

x_train = reshape_img(x_train)
x_test = reshape_img(x_test)


def get_number_datapoints(number1, number2):

	x = x_train[np.isin(y_train,[number1,number2]),:]
	y = y_train[np.isin(y_train,[number1,number2])]

	xs = x_test[np.isin(y_test,[number1,number2]),:]
	ys = y_test[np.isin(y_test,[number1,number2])]
	return (x, y), (xs, ys)

# pick out 1 vs. 6

(x, y), (xs, ys) = get_number_datapoints(1, 6)

model = ClassificationModel()
model.fit(x, y)

m, v, ms = model.predict(xs)

# model is extremely slow - reduce layers and num_inducing
posterior_samples = ms

# save the data and posterior results
ms = ms.reshape((ms.shape[0],ms.shape[1]))
np.savetxt('MNIST_1_6_posterior.txt', ms)
np.savetxt('MNIST_1_6_data.txt', xs)

##############################################################

# pick out 1 vs 7
(x, y), (xs, ys) = get_number_datapoints(1, 8)

model = ClassificationModel()
model.fit(xs, ys)

m, v, ms = model.predict(xs)

# model is extremely slow - reduce layers and num_inducing
posterior_samples = ms

# save the data and posterior results
ms = ms.reshape((ms.shape[0],ms.shape[1]))
np.savetxt('MNIST_1_8_posterior.txt', ms)
np.savetxt('MNIST_1_8_data.txt', xs)

##############################################################

# pick out 1 vs 7
(x, y), (xs, ys) = get_number_datapoints(1, 0)

model = ClassificationModel()
model.fit(xs, ys)

m, v, ms = model.predict(xs)

# model is extremely slow - reduce layers and num_inducing
posterior_samples = ms

# save the data and posterior results
ms = ms.reshape((ms.shape[0],ms.shape[1]))
np.savetxt('MNIST_1_0_posterior.txt', ms)
np.savetxt('MNIST_1_0_data.txt', xs)
