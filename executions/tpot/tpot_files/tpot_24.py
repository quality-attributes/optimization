# Best pipeline: MultinomialNB(VarianceThreshold(RandomForestClassifier(SelectPercentile(input_matrix, percentile=95), bootstrap=False, criterion=gini, max_features=0.25, min_samples_leaf=1, min_samples_split=19, n_estimators=100), threshold=0.005), alpha=0.1,True)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6734901960784313
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=95),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.25, min_samples_leaf=1, min_samples_split=19, n_estimators=100)),
    VarianceThreshold(threshold=0.005),
    MultinomialNB(alpha=0.1,True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)