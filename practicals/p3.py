from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

data = np.genfromtxt('https://raw.githubusercontent.com/Tan-Moy/medium_articles/master/art2_multivariate_linear_regression/home.txt', delimiter=',')
data = normalize(data, axis=0)

#splits featuers and labels
X = data[:, 0:2]
Y = data[:, 2:]


#hyperpermitors
learning_rate = 0.09
max_iteration = 500

s_learning_rate = 0.06
s_max_iteration = 500

mb_learning_rate = 0.09
mb_max_iteration = 500
batch_size = 16


#theta peramitors
theta = np.zeros((data.shape[1], 1))
s_theta = np.zeros((data.shape[1], 1))
mb_theta = np.zeros((data.shape[1], 1))

#hypothosis funtion
def h (theta, X) :
  tempX = np.ones((X.shape[0], X.shape[1] + 1))
  tempX[:,1:] = X
  return np.matmul(tempX, theta)


#loss funtion
def loss (theta, X, Y) :
  return np.average(np.square(Y - h(theta, X))) / 2

#calculate gradient
def gradient (theta, X, Y) :
  tempX = np.ones((X.shape[0], X.shape[1] + 1))
  tempX[:,1:] = X
  d_theta = - np.average((Y - h(theta, X)) * tempX, axis= 0)
  d_theta = d_theta.reshape((d_theta.shape[0], 1))
  return d_theta


#types of gradient decent already coverd
def gradient_descent (theta, X, Y, learning_rate, max_iteration, gap) :
  cost = np.zeros(max_iteration)
  for i in range(max_iteration) :
    d_theta = gradient(theta, X, Y)
    theta = theta - learning_rate * d_theta
    cost[i] = loss(theta, X, Y)
    if i % gap == 0 :
      print ('iteration : ', i, ' loss : ', loss(theta, X, Y))
  return theta, cost


def minibatch_gradient_descent(theta, X, Y, learning_rate, max_iteration, batch_size, gap):
    cost = np.zeros(max_iteration)
    for i in range(max_iteration):
        for j in range(0, X.shape[0], batch_size):
            d_theta = gradient(theta, X[j:j + batch_size, :], Y[j:j + batch_size, :])
            theta = theta - learning_rate * d_theta

        cost[i] = loss(theta, X, Y)
        if i % gap == 0:
            print('iteration : ', i, ' loss : ', loss(theta, X, Y))
    return theta, cost


def stochastic_gradient_descent(theta, X, Y, learning_rate, max_iteration, gap):
    cost = np.zeros(max_iteration)
    for i in range(max_iteration):
        for j in range(X.shape[0]):
            d_theta = gradient(theta, X[j, :].reshape(1, X.shape[1]), Y[j, :].reshape(1, 1))
            theta = theta - learning_rate * d_theta

        cost[i] = loss(theta, X, Y)
        if i % gap == 0:
            print('iteration : ', i, ' loss : ', loss(theta, X, Y))
    return theta, cost


theta, cost = gradient_descent (theta, X, Y, learning_rate, max_iteration, 100)
#s_theta, s_cost = stochastic_gradient_descent (s_theta, X, Y, s_learning_rate, s_max_iteration, 100)
#mb_theta, mb_cost = minibatch_gradient_descent (mb_theta, X, Y, mb_learning_rate, mb_max_iteration, batch_size, 100)