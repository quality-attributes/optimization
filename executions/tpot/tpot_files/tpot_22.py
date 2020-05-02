# Best pipeline: ExtraTreesClassifier(CombineDFs(RFE(XGBClassifier(Normalizer(BernoulliNB(input_matrix, alpha=1.0,True), norm=l2), learning_rate=0.1, max_depth=8, min_child_weight=9, n_estimators=100, nthread=1, subsample=0.8500000000000001), criterion=entropy, max_features=0.2, n_estimators=100, step=0.1), input_matrix), bootstrap=True, criterion=gini, max_features=0.35000000000000003, min_samples_leaf=7, min_samples_split=3, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6851764705882353
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
            Normalizer(norm="l2"),
            StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=8, min_child_weight=9, n_estimators=100, nthread=1, subsample=0.8500000000000001)),
            RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.2, n_estimators=100), step=0.1)
        ),
        FunctionTransformer(copy)
    ),
    ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.35000000000000003, min_samples_leaf=7, min_samples_split=3, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)