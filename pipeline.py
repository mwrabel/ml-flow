from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://www.tomasbeuzen.com/post/scikit-learn-gridsearch-pipelines/
# pd.options.plotting.backend = "plotly"

X, y = make_classification(n_samples=1000,
                           n_features=30,
                           n_informative=5,
                           n_redundant=5,
                           n_classes=2,
                           random_state=123)


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(mutual_info_classif, k=5)),
    ('classifier', LogisticRegression())
])


search_space = [
    {
        'selector__k': [5, 10, 20, 30]
    },
    {
        'classifier': [LogisticRegression(solver='lbfgs')],
        'classifier__C': [0.01, 0.1, 1.0]
    },
    {
        'classifier': [RandomForestClassifier(n_estimators=100)],
        'classifier__max_depth': [5, 10, None]
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 7, 11],
        'classifier__weights': ['uniform', 'distance']
    }
]

clf = GridSearchCV(pipe, search_space, cv=10, verbose=0)
clf = clf.fit(X, y)

clf.best_estimator_
