# Best pipeline: ExtraTreesClassifier(LinearSVC(BernoulliNB(ExtraTreesClassifier(input_matrix, bootstrap=True, criterion=gini, max_features=0.6000000000000001, min_samples_leaf=5, min_samples_split=12, n_estimators=100), alpha=1.0,True), C=0.5, dual=True, loss=hinge, penalty=l2, tol=1e-05), bootstrap=True, criterion=entropy, max_features=0.8500000000000001, min_samples_leaf=8, min_samples_split=13, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6853333333333333
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.6000000000000001, min_samples_leaf=5, min_samples_split=12, n_estimators=100)),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
    StackingEstimator(estimator=LinearSVC(C=0.5, dual=True, loss="hinge", penalty="l2", tol=1e-05)),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.8500000000000001, min_samples_leaf=8, min_samples_split=13, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)