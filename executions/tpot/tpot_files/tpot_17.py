# Best pipeline: MultinomialNB(XGBClassifier(ExtraTreesClassifier(input_matrix, bootstrap=True, criterion=gini, max_features=0.45, min_samples_leaf=4, min_samples_split=11, n_estimators=100), learning_rate=0.1, max_depth=9, min_child_weight=6, n_estimators=100, nthread=1, subsample=0.25), alpha=0.1,True)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6814117647058824
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=4, min_samples_split=11, n_estimators=100)),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=9, min_child_weight=6, n_estimators=100, nthread=1, subsample=0.25)),
    MultinomialNB(alpha=0.1,True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)