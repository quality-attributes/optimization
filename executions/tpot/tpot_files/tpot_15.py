# Best pipeline: RandomForestClassifier(GaussianNB(Normalizer(SelectPercentile(BernoulliNB(input_matrix, alpha=1.0,True), percentile=38), norm=l1)), bootstrap=True, criterion=gini, max_features=0.3, min_samples_leaf=6, min_samples_split=13, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6775686274509803
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
    SelectPercentile(score_func=f_classif, percentile=38),
    Normalizer(norm="l1"),
    StackingEstimator(estimator=GaussianNB()),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.3, min_samples_leaf=6, min_samples_split=13, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)