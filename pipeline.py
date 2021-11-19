from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import make_classification, fetch_openml
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

from scipy.stats import uniform, randint, loguniform

import pandas as pd


# https://www.tomasbeuzen.com/post/scikit-learn-gridsearch-pipelines/
# pd.options.plotting.backend = "plotly"

# X, y = make_classification(n_samples=5000,
#                            n_features=30,
#                            n_informative=10,
#                            n_redundant=5,
#                            n_repeated=5,
#                            flip_y=0.1,
#                            n_classes=2,
#                            class_sep=0.01,
#                            random_state=1)

# Load data from https://www.openml.org/d/40945
X, y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True)

X = pd.DataFrame(X)
y = pd.Series(y)

#X = X[['name']]

# Split to train / test
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.20, random_state=42)

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric",
         Pipeline(steps=[
             ("imputer", SimpleImputer(strategy='mean')),
             ("scaler", StandardScaler(with_mean=True, with_std=True))
         ]),
         make_column_selector(dtype_include=['float', 'int'])),
        ("category",
         Pipeline(steps=[
             ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
             ("encoder", OneHotEncoder(handle_unknown="ignore"))
         ]),
         make_column_selector(dtype_include='category')),
        ("high_cardinality",
         Pipeline(steps=[
             ("imputer", SimpleImputer(strategy='constant', fill_value='missing', missing_values=None)),
             ("hasher", FeatureHasher(n_features=10, input_type='string'))
         ]),
         make_column_selector(dtype_include='object'),
         )
    ], remainder='passthrough'
)

#xd = preprocessor.fit_transform(X, y)

# Classification Pipeline
classifier = Pipeline(steps=[
    ('poly', PolynomialFeatures()),
    ('reductor', PCA()),
    ('selector', SelectFromModel(ExtraTreesClassifier())),
    ('estimator', RandomForestClassifier())
])

# Main Pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Search space
common_search_space = {
    'classifier__poly__interaction_only': [True, False],
    'classifier__reductor__n_components': randint(5, 15),
    'classifier__selector__threshold': ['0.125*mean', '0.25*mean', '0.5*mean', '0.75*mean', '1*mean', '1.25*mean'],
}

search_space = [
    # {
    #     **common_search_space,
    #     'classifier__estimator': [LogisticRegression(solver='saga')],
    #     'classifier__estimator__C': uniform(loc=0.00, scale=1.00),
    #     'classifier__estimator__penalty': ['none', 'l1', 'l2', 'elasticnet']
    # },
    {
        **common_search_space,
        'classifier__estimator': [RandomForestClassifier()],
        'classifier__estimator__n_estimators': randint(10, 500),
        'classifier__estimator__criterion': ['gini', 'entropy'],
        'classifier__estimator__max_depth': randint(5, 16),
        'classifier__estimator__min_samples_split': uniform(0.0001, 0.009),
        'classifier__estimator__min_samples_leaf': uniform(0.00005, 0.0095),
        'classifier__estimator__min_weight_fraction_leaf': uniform(loc=0.00, scale=0.05),
        'classifier__estimator__max_features': ['sqrt', 'log2'],
        'classifier__estimator__min_impurity_decrease': uniform(loc=0.00, scale=0.1),
        'classifier__estimator__ccp_alpha': loguniform(a=0.000000001, b=1.0),
        'classifier__estimator__max_samples': uniform(0.1, 0.9)
    },
    {
        **common_search_space,
        'classifier__estimator': [KNeighborsClassifier()],
        'classifier__estimator__n_neighbors': randint(3, 12),  # max 11
        'classifier__estimator__weights': ['uniform', 'distance'],
        'classifier__estimator__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'classifier__estimator__leaf_size': randint(10, 100),
        'classifier__estimator__p': randint(1, 3),
        'classifier__estimator__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }
]

fold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# clf = GridSearchCV(pipe, search_space, cv=10, verbose=0)
clf = RandomizedSearchCV(pipe, search_space, cv=fold, scoring='roc_auc', n_iter=10, random_state=1, n_jobs=-1, verbose=2)
# clf = HalvingRandomSearchCV(pipe, search_space, cv=fold, verbose=1, scoring='roc_auc', n_jobs=-1,
#                             aggressive_elimination=False, factor=2, min_resources=50)

clf = clf.fit(train_features, train_labels)

score_auc_train = roc_auc_score(train_labels, clf.predict_proba(train_features)[:, 1])
score_auc_test = roc_auc_score(test_labels, clf.predict_proba(test_features)[:, 1])
print(clf.best_estimator_)

print(f'training auc: {score_auc_train}')
print(f'testing auc: {score_auc_test}')

train_predictions = clf.predict(train_features)
print(classification_report(train_labels, train_predictions))

test_predictions = clf.predict(test_features)
print(classification_report(test_labels, test_predictions))
