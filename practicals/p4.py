import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression

#boring 2d graph
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
#plt.scatter(x, y);


def findFit(x, y):
    """
    trains alin model on the data
    then creates lots of points close together and puts them though the prediction model
    returns these many points that form a line
    :param x:
    :param y:
    :return:
    """

    model = LinearRegression(fit_intercept=True)

    model.fit(x[:, np.newaxis], y)

    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    print("Model slope:    ", model.coef_[0])
    print("Model intercept:", model.intercept_)

    return xfit, yfit

print(findFit(x, y))


#multi dimentisonal data
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
#multi dimentsional lineaer eqn
y = 0.5 + np.dot(X, [1.5, -2., 1.])


#model = LinearRegression(fit_intercept=True)
#model.fit(X, y)
#print(model.intercept_)
#print(model.coef_)


#basis function regression:
"""
takes the L:
y=a0+a1x1+a2x2+a3x3+⋯
and sets xs equal to some function:
example: fn(x)=xn 
y=a0+a1x+a2**2+a3**3+⋯

this specific one called polynomical features in sk learn
"""


from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])

#this turesn our x into a 2d array of the squares of the 2,3,4
"""
array([[ 2.,  4.,  8.],
       [ 3.,  9., 27.],
       [ 4., 16., 64.]])
"""


#to do this and then fir to model we make a pipline!

from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

#example:
"""
this noisy sinewave data get fit well with a polynomial feature vector of up to degree 7
"""
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

#out line of mny equaly spaced points
xfit = np.linspace(0, 10, 1000)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])


#can do with any function not just polnomials

#gausan as basis funciton:

from sklearn.base import BaseEstimator, TransformerMixin


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)


gauss_model = make_pipeline(GaussianFeatures(20),
                            LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])


#regularisation
"""
beings the size of values of thetas into the cost fucntion
larger thetas effect the midoel more, and can cause over fitting
"""
#testing bit with overfitting model from a crazy basis function that doent help at all:
def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))

    if title:
        ax[0].set_title(title)

    ax[1].plot(model.steps[0][1].centers_,
               model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location',
              ylabel='coefficient',
              xlim=(0, 10))

#this gausian funciton of 30 compleatly screws up the model
model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)


#ridge regression:

from sklearn.linear_model import Ridge

#here alpha peram controlls how close the thetas can get to the optimal ones that will defianly overfit the model
#smaller alpha the more general it gets. greater than 1 the closer to optimal thetas of overfitting
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
basis_plot(model, title='Ridge Regression')


#lasso regression:
"""
like to set peramiters to 0 entirely 
tends to favor sparser models
"""

from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
basis_plot(model, title='Lasso Regression')










