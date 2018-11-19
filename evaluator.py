
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

class ModelComparer:
    def __init__(self, x, y):
        self.LR = ('LR', LogisticRegression())
        self.LDA = ('LDA', LinearDiscriminantAnalysis())
        self.KNN = ('KNN', KNeighborsClassifier())
        self.CART = ('CART', DecisionTreeClassifier())
        self.GNB = ('NB', GaussianNB())
        self.SVM = ('SVC', SVC())
        self.models = [self.LR, self.LDA, self.KNN, self.CART, self.GNB, self.SVM]
        self.x = x
        self.y = y

    def runCrossValidation(self, splits, scoring):
        seed = 7
        results = []
        names = []
        scoring = 'accuracy'
        for name, model in self.models:
            kfold = model_selection.KFold(n_splits=splits, random_state=seed)
            cv_results = model_selection.cross_val_score(model, self.x, self.y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        return results, names
