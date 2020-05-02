# Best pipeline: ExtraTreesClassifier(BernoulliNB(ExtraTreesClassifier(SGDClassifier(GradientBoostingClassifier(KNeighborsClassifier(input_matrix, n_neighbors=25, p=2, weights=uniform), learning_rate=0.01, max_depth=6, max_features=0.8, min_samples_leaf=18, min_samples_split=15, n_estimators=100, subsample=0.1), alpha=0.01, eta0=1.0, fit_intercept=True, l1_ratio=0.75, learning_rate=invscaling, loss=hinge, penalty=elasticnet, power_t=10.0), bootstrap=True, criterion=gini, max_features=0.55, min_samples_leaf=11, min_samples_split=7, n_estimators=100), alpha=1.0,False), bootstrap=True, criterion=entropy, max_features=0.8500000000000001, min_samples_leaf=3, min_samples_split=13, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6850980392156863
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=25, p=2, weights="uniform")),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.01, max_depth=6, max_features=0.8, min_samples_leaf=18, min_samples_split=15, n_estimators=100, subsample=0.1)),
    StackingEstimator(estimator=SGDClassifier(alpha=0.01, eta0=1.0, fit_intercept=True, l1_ratio=0.75, learning_rate="invscaling", loss="hinge", penalty="elasticnet", power_t=10.0)),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.55, min_samples_leaf=11, min_samples_split=7, n_estimators=100)),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,False)),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.8500000000000001, min_samples_leaf=3, min_samples_split=13, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)