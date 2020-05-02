# Best pipeline: ExtraTreesClassifier(CombineDFs(BernoulliNB(SGDClassifier(input_matrix, alpha=0.001, eta0=1.0, fit_intercept=True, l1_ratio=0.5, learning_rate=invscaling, loss=squared_hinge, penalty=elasticnet, power_t=10.0), alpha=1.0,True), input_matrix), bootstrap=False, criterion=gini, max_features=0.1, min_samples_leaf=1, min_samples_split=12, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6655686274509804
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=make_pipeline(
            StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=1.0, fit_intercept=True, l1_ratio=0.5, learning_rate="invscaling", loss="squared_hinge", penalty="elasticnet", power_t=10.0)),
            BernoulliNB(alpha=1.0,True)
        )),
        FunctionTransformer(copy)
    ),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.1, min_samples_leaf=1, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)