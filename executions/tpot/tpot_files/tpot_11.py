# Best pipeline: LinearSVC(XGBClassifier(DecisionTreeClassifier(VarianceThreshold(input_matrix, threshold=0.005), criterion=gini, max_depth=2, min_samples_leaf=20, min_samples_split=8), learning_rate=0.001, max_depth=2, min_child_weight=7, n_estimators=100, nthread=1, subsample=0.4), C=0.5, dual=False, loss=squared_hinge, penalty=l2, tol=0.1)
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6773333333333333
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.005),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_leaf=20, min_samples_split=8)),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.001, max_depth=2, min_child_weight=7, n_estimators=100, nthread=1, subsample=0.4)),
    LinearSVC(C=0.5, dual=False, loss="squared_hinge", penalty="l2", tol=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)