# Best pipeline: ExtraTreesClassifier(BernoulliNB(XGBClassifier(input_matrix, learning_rate=0.001, max_depth=3, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.3), alpha=1.0,True), bootstrap=True, criterion=gini, max_features=0.1, min_samples_leaf=1, min_samples_split=18, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6774901960784314
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.001, max_depth=3, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.3)),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
    ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.1, min_samples_leaf=1, min_samples_split=18, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)