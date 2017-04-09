#!/usr/bin/python


import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def plot_decision_boundary(Classifier,X,Plot=plt,Alpha=1):
	
	x1_min, x1_max = X[:, 0].min() - .7, X[:, 0].max() + .7
	x1 = np.linspace(x1_min, x1_max, 100).reshape(-1, 1)

	beta0,beta1,beta2 = Classifier.intercept_[0],Classifier.coef_[0,0],Classifier.coef_[0,1]
	x2 = (-beta0 - beta1*x1)/beta2

	#Plot the boudary
	Plot.plot(x1,x2,"-k",label="Boundary",alpha=Alpha)
	


def surface_plot(Classifier,X,y,Plot=plt,boundary=False,offset=0.7):

	x1_min, x1_max = X[:, 0].min() - offset, X[:, 0].max() + offset
	x2_min, x2_max = X[:, 1].min() - offset, X[:, 1].max() + offset
	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

	Plot.xlim(xx1.min(), xx1.max())
	Plot.ylim(xx2.min(), xx2.max())
	
	ax = Plot.gca()
	Z = Classifier.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, 1]
	Z = Z.reshape(xx1.shape)
	cs = ax.contourf(xx1, xx2, Z, cmap='RdBu_r', alpha=.3)
	cs2 = ax.contour(xx1, xx2, Z, cmap='RdBu_r', alpha=.3)
	Plot.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)
	
	Plot.scatter(X[:, 0], X[:, 1], c=y, lw=0, s=40)
	
	#Plot the boudary
	if boundary: 
		plot_decision_boundary(Classifier,X,Plot)
		
def plot_c_boundary(Classifier,X,Plot=plt):

	beta00 = Classifier.coef_[0,0]
	beta10 = Classifier.coef_[0,1]
	beta01 = Classifier.coef_[0,2]
	beta20 = Classifier.coef_[0,3]
	beta11 = Classifier.coef_[0,4]
	beta02 = Classifier.coef_[0,5]
	beta0 = Classifier.intercept_

	x1_min, x1_max = X[:, 0].min() - .2, X[:, 0].max() + .2
	x2_min, x2_max = X[:, 1].min() - .2, X[:, 1].max() + .2
	x1 = np.linspace(x1_min, x1_max, 1000).reshape(-1, 1)
	x2 = np.linspace(x2_min, x2_max, 1000).reshape(-1, 1)

	
	X1,X2 = np.meshgrid(x1,x2)
	F = beta0 + beta00 + beta10*X1 + beta01*X2 + beta20*X1**2 + beta11*X1*X2 + beta02*X2**2
	Plot.contour(X1,X2,F,[0],label="Boundary",linewidths=3)
	





	
	