from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV, train_test_split, StratifiedKFold
import numpy as np
from scipy.stats import randint, uniform
from sklearn.metrics import roc_auc_score

from sklearn.datasets import fetch_kddcup99
# Load data
df = load_breast_cancer(return_X_y=True, as_frame=True)
x = df[0]
y = df[1]
del df

# x, y = make_classification(n_samples=5000, n_features=30, n_informative=10, n_redundant=5, n_repeated=5, flip_y=0.1, n_classes=2, class_sep=0.01, random_state=10)



# Split to train / test
train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit out of the box RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10)
rf.fit(train_features, train_labels)
# Score model
score_rf = roc_auc_score(test_labels, rf.predict_proba(test_features)[:, 1])
print(f'ROC AUC Score for out of the box model: {score_rf}')

# Build pipeline
pipe = Pipeline([
    # ('scaler', StandardScaler()),
    # ('selector', SelectKBest(mutual_info_classif, k=5)),
    ('classifier', RandomForestClassifier())
])

# Build CV folds
cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

# Declare param grid
param_grid = {
    # 'selector__k': randint(1, x.shape[1]+1),
    'classifier__n_estimators': randint(10, 500),
    # 'classifier__max_samples': uniform(0.1, 0.9) # 0.1 -> 0.9
    # 'classifier__criterion': ['gini', 'entropy'],
    # 'classifier__max_depth': randint(10, 20),
    # 'classifier__min_samples_split': randint(2, 4),
    # 'classifier__min_samples_leaf': randint(1, 2),
    # 'classifier__min_weight_fraction_leaf': uniform(loc=0.00, scale=0.1),
    # 'classifier__max_features': ['sqrt', 'log2'],
    # 'classifier__min_impurity_decrease': uniform(loc=0.00, scale=0.1),
}

# Fit rf model with RandomizedSearchCV
clf = RandomizedSearchCV(pipe, param_grid, cv=cv, verbose=2, n_iter=5, scoring='roc_auc', n_jobs=-1)
clf = clf.fit(train_features, train_labels)
# Score model
score_randomized = roc_auc_score(test_labels, clf.predict_proba(test_features)[:, 1])
print(f'ROC AUC Score for RandomizedSearchCV model: {score_randomized}')
print(clf.best_params_)

# Fit rf model with HalvingRandomSearchCV
clf_halving = HalvingRandomSearchCV(pipe, param_grid, cv=cv, verbose=1, scoring='roc_auc', n_jobs=-1,
                                    aggressive_elimination=True, factor=2, min_resources=20)
clf_halving = clf_halving.fit(train_features, train_labels)
# Score model
score_halving = roc_auc_score(test_labels, clf_halving.predict_proba(test_features)[:, 1])
print(f'ROC AUC Score for HalvingRandomSearchCV model: {score_halving}')
print(clf_halving.best_params_)

print(f'ROC AUC Score for out of the box model: {score_rf}')
print(f'ROC AUC Score for RandomizedSearchCV model: {score_randomized}')
print(f'ROC AUC Score for HalvingRandomSearchCV model: {score_halving}')
