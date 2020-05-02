# Best pipeline: MultinomialNB(BernoulliNB(ExtraTreesClassifier(ZeroCount(RandomForestClassifier(SelectFromModel(input_matrix, criterion=gini, max_features=0.7000000000000001, n_estimators=100, threshold=0.0), bootstrap=False, criterion=gini, max_features=0.25, min_samples_leaf=1, min_samples_split=15, n_estimators=100)), bootstrap=True, criterion=gini, max_features=0.1, min_samples_leaf=20, min_samples_split=19, n_estimators=100), alpha=1.0,True), alpha=0.1,True)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6971764705882354
exported_pipeline = make_pipeline(
    SelectFromModel(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.7000000000000001, n_estimators=100), threshold=0.0),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.25, min_samples_leaf=1, min_samples_split=15, n_estimators=100)),
    ZeroCount(),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.1, min_samples_leaf=20, min_samples_split=19, n_estimators=100)),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
    MultinomialNB(alpha=0.1,True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)