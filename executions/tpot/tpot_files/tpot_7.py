# Best pipeline: GradientBoostingClassifier(ZeroCount(BernoulliNB(DecisionTreeClassifier(XGBClassifier(VarianceThreshold(input_matrix, threshold=0.0005), learning_rate=0.1, max_depth=1, min_child_weight=17, n_estimators=100, nthread=1, subsample=0.6000000000000001), criterion=gini, max_depth=2, min_samples_leaf=19, min_samples_split=14), alpha=1.0,True)), learning_rate=0.01, max_depth=4, max_features=0.8500000000000001, min_samples_leaf=6, min_samples_split=15, n_estimators=100, subsample=0.6500000000000001)
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6933333333333334
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.0005),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=1, min_child_weight=17, n_estimators=100, nthread=1, subsample=0.6000000000000001)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_leaf=19, min_samples_split=14)),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
    ZeroCount(),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=4, max_features=0.8500000000000001, min_samples_leaf=6, min_samples_split=15, n_estimators=100, subsample=0.6500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)