# Best pipeline: RandomForestClassifier(MultinomialNB(CombineDFs(KNeighborsClassifier(input_matrix, n_neighbors=23, p=1, weights=distance), VarianceThreshold(MultinomialNB(ExtraTreesClassifier(input_matrix, bootstrap=False, criterion=gini, max_features=0.9500000000000001, min_samples_leaf=12, min_samples_split=16, n_estimators=100), alpha=10.0,True), threshold=0.01)), alpha=0.1,False), bootstrap=True, criterion=gini, max_features=0.1, min_samples_leaf=4, min_samples_split=14, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6814117647058824
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=23, p=1, weights="distance")),
        make_pipeline(
            StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9500000000000001, min_samples_leaf=12, min_samples_split=16, n_estimators=100)),
            StackingEstimator(estimator=MultinomialNB(alpha=10.0,True)),
            VarianceThreshold(threshold=0.01)
        )
    ),
    StackingEstimator(estimator=MultinomialNB(alpha=0.1,False)),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.1, min_samples_leaf=4, min_samples_split=14, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)