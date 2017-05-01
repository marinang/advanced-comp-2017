#!/usr/bin/python
import csv
import subprocess
from tempfile import mkstemp
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from scipy import interp

from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc

from IPython.core.display import HTML


def load_wine(data_dir='../data'):
	X = []
	y = []

	red = csv.reader(open(data_dir + '/winequality-red.csv'),
					 delimiter=';', quotechar='"')
	# skip header
	next(red)
	for l in red:
		y.append(float(l[-1]))
		X.append(list(map(float, l[:-1])))
	white = csv.reader(open(data_dir + '/winequality-white.csv'),
					   delimiter=';', quotechar='"')
	# skip header
	next(white)
	for l in white:
		y.append(float(l[-1]))
		X.append(list(map(float, l[:-1])))
	X = np.array(X)
	y = np.array(y)
	return X, y


def plot_surface(clf, X, y, n_steps=250, show=True, ylim=None,
				 xlim=None, ax=None, size=24, proba = True):
	if ax is None:
		fig = plt.figure()
		ax = plt.gca()

	if xlim is None:
		xlim = X[:, 0].min(), X[:, 0].max()
	if ylim is None:
		ylim = X[:, 1].min(), X[:, 1].max()
	xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps),
						 np.linspace(ylim[0], ylim[1], n_steps))
#
	if proba:		
		if hasattr(clf, "decision_function"):
			z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		else:
			z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
	else:	
		z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	z = z.reshape(xx.shape)
	
	if proba:
		ax.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
	else:
		ax.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.coolwarm)

	ax.scatter(X[:, 0], X[:, 1], c=y, s=size)
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)

	if show:
		plt.show()


def draw_tree(clf, feature_names, **kwargs):
	_, name = mkstemp(suffix='.dot')
	_, svg_name = mkstemp(suffix='.svg')
	export_graphviz(clf, out_file=name,
					feature_names=feature_names,
					**kwargs)
	command = ["dot", "-Tsvg", name, "-o", svg_name]
	subprocess.check_call(command)
	return HTML(open(svg_name).read())


# Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
def draw_roc_curve(classifier, cv, X, y):
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
	lw = 2

	for i, (train, test) in enumerate(cv):
		probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
			 label='Luck')

	mean_tpr /= len(cv)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
			 label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()


def plot_loss(est, X_test, y_test, ax=None, label='',
			  train_color=None, test_color=None, alpha=1.0, ylim=(0, 0.7)):
	n_estimators = len(est.estimators_)
	test_dev = np.empty(n_estimators)

	for i, pred in enumerate(est.staged_predict(X_test)):
		test_dev[i] = est.loss_(y_test, pred)

	if ax is None:
		fig = plt.figure()
		ax = plt.gca()

	ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color,
			label='Test %s' % label, linewidth=2, alpha=alpha)
	ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color,
			label='Train %s' % label, linewidth=2, alpha=alpha)
	ax.set_ylabel('Loss')
	ax.set_xlabel('n_estimators')
	ax.set_ylim(ylim)
	return test_dev, ax


def plot_decision_boundary(Classifier,X,Plot=plt,Alpha=1):
	
	x1_min, x1_max = X[:, 0].min() - .7, X[:, 0].max() + .7
	x1 = np.linspace(x1_min, x1_max, 100).reshape(-1, 1)

	beta0,beta1,beta2 = Classifier.intercept_[0],Classifier.coef_[0,0],Classifier.coef_[0,1]
	x2 = (-beta0 - beta1*x1)/beta2

	#Plot the boudary
	Plot.plot(x1,x2,"-k",label="Boundary",alpha=Alpha)
	


def surface_plot_m(Classifier,X,y,Plot=plt,boundary=False,offset=0.7):

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
	

	





	
	