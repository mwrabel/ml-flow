from sklearn.datasets import make_classification

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import numpy as np

# generate dataset
X,y = make_classification(
    n_samples=10000, n_features=20, n_informative=2, n_redundant=0, n_repeated=2, n_classes=2, n_clusters_per_class=1,
    weights=[0.998], flip_y=0.00, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=1
)

# define model
model = SVC(gamma='scale')
# model = RandomForestClassifier()
# wrap the model
model_calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)  # method='sigmoid'
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
# evaluate model
scores = cross_val_score(model_calibrated, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % np.mean(scores))


# define pipeline
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

pipe = Pipeline([
    ('model', CalibratedClassifierCV(base_estimator=SVC(), method='isotonic', cv=5))
])

params = {
    'model__base_estimator__C': [0.5, 1, 2],
    'model__base_estimator__kernel': ['poly', 'rbf', 'sigmoid'],
    'model__base_estimator__gamma': ['scale', 'auto'],
}

fold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

model = RandomizedSearchCV(pipe, params, scoring='roc_auc', cv=fold, n_iter=10, random_state=1, n_jobs=-1, verbose=2)
model.fit(X, y)

model.best_score_