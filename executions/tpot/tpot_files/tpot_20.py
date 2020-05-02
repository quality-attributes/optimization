# Best pipeline: RandomForestClassifier(XGBClassifier(BernoulliNB(DecisionTreeClassifier(XGBClassifier(input_matrix, learning_rate=0.5, max_depth=9, min_child_weight=10, n_estimators=100, nthread=1, subsample=0.05), criterion=gini, max_depth=2, min_samples_leaf=16, min_samples_split=13), alpha=1.0,True), learning_rate=0.01, max_depth=3, min_child_weight=2, n_estimators=100, nthread=1, subsample=0.55), bootstrap=True, criterion=gini, max_features=0.1, min_samples_leaf=1, min_samples_split=12, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6814117647058824
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.5, max_depth=9, min_child_weight=10, n_estimators=100, nthread=1, subsample=0.05)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_leaf=16, min_samples_split=13)),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.01, max_depth=3, min_child_weight=2, n_estimators=100, nthread=1, subsample=0.55)),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.1, min_samples_leaf=1, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)