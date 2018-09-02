# Support Vector Machine Classifier (w. RBF kernel)
# Basic ML algorithm to be tested on small datasets (< 10k samples)

import numpy as np
import pandas as pd
from sklearn.svm import SVC

#Import data as pandas dataset named "X"(to be done)

# (...)

X=X.drop('name', axis=1)
X.dropna(axis=1, how='all')
y=X['status']
X=X.drop('status', axis=1)

#Split data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

#Standard preprocessing

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#ML procedure
#Outliers: Isomap (n_neighbors from 2 to 5 and n_components from 4 to 6)

for n_neighbors in range(2,6):
    for n_components in range(4,7):
        from sklearn.manifold import Isomap
        iso=Isomap(n_neighbors=n_neighbors, n_components=n_components)
        iso.fit(X_train)
        X_train=iso.transform(X_train)
        X_test=iso.transform(X_test)


        def SVC_model(C,gamma) :
            model = SVC(C=C, gamma=gamma)
            model.fit(X_train, y_train)
            score=model.score(X_test, y_test)
            return score

        score=0
        best_score=0

        for C in np.arange(0.05,2.05,0.05):
            for gamma in np.arange(0.001,0.101,0.001):
                score=SVC_model(C,gamma)
                if score > best_score:
                    best_score=score

print(best_score)

#ML NOT optimized
#This simple procedure to be used for preliminary testing purposes only