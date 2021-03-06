# Best pipeline: SGDClassifier(OneHotEncoder(ExtraTreesClassifier(XGBClassifier(BernoulliNB(ExtraTreesClassifier(input_matrix, bootstrap=True, criterion=entropy, max_features=0.35000000000000003, min_samples_leaf=15, min_samples_split=18, n_estimators=100), alpha=1.0,True), learning_rate=0.5, max_depth=2, min_child_weight=13, n_estimators=100, nthread=1, subsample=0.25), bootstrap=True, criterion=entropy, max_features=0.6500000000000001, min_samples_leaf=3, min_samples_split=4, n_estimators=100), minimum_fraction=0.2, sparse=False, threshold=10), alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=1.0, learning_rate=constant, loss=hinge, penalty=elasticnet, power_t=1.0)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6734117647058824
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=15, min_samples_split=18, n_estimators=100)),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.5, max_depth=2, min_child_weight=13, n_estimators=100, nthread=1, subsample=0.25)),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.6500000000000001, min_samples_leaf=3, min_samples_split=4, n_estimators=100)),
    OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    SGDClassifier(alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=1.0, learning_rate="constant", loss="hinge", penalty="elasticnet", power_t=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)