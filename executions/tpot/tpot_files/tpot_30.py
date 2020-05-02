# Best pipeline: ExtraTreesClassifier(BernoulliNB(OneHotEncoder(SelectPercentile(Binarizer(input_matrix, threshold=0.15000000000000002), percentile=59), minimum_fraction=0.1, sparse=False, threshold=10), alpha=1.0,False), bootstrap=True, criterion=gini, max_features=0.45, min_samples_leaf=2, min_samples_split=15, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6891764705882354
exported_pipeline = make_pipeline(
    Binarizer(threshold=0.15000000000000002),
    SelectPercentile(score_func=f_classif, percentile=59),
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,False)),
    ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=2, min_samples_split=15, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)