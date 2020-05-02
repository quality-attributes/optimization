# Best pipeline: MultinomialNB(GaussianNB(PolynomialFeatures(VarianceThreshold(ExtraTreesClassifier(KNeighborsClassifier(input_matrix, n_neighbors=21, p=1, weights=uniform), bootstrap=True, criterion=entropy, max_features=0.35000000000000003, min_samples_leaf=3, min_samples_split=10, n_estimators=100), threshold=0.005), degree=2, include_bias=False, interaction_only=False)), alpha=0.1,True)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6934901960784312
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=21, p=1, weights="uniform")),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=3, min_samples_split=10, n_estimators=100)),
    VarianceThreshold(threshold=0.005),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=GaussianNB()),
    MultinomialNB(alpha=0.1,True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)