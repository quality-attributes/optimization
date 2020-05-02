# Best pipeline: KNeighborsClassifier(Normalizer(MultinomialNB(Normalizer(DecisionTreeClassifier(input_matrix, criterion=gini, max_depth=1, min_samples_leaf=20, min_samples_split=10), norm=max), alpha=10.0,False), norm=max), n_neighbors=25, p=2, weights=uniform)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6614901960784314
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=1, min_samples_leaf=20, min_samples_split=10)),
    Normalizer(norm="max"),
    StackingEstimator(estimator=MultinomialNB(alpha=10.0,False)),
    Normalizer(norm="max"),
    KNeighborsClassifier(n_neighbors=25, p=2, weights="uniform")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)