# Best pipeline: GradientBoostingClassifier(Binarizer(BernoulliNB(GradientBoostingClassifier(input_matrix, learning_rate=0.1, max_depth=1, max_features=0.4, min_samples_leaf=12, min_samples_split=11, n_estimators=100, subsample=0.9000000000000001), alpha=0.1,True), threshold=0.1), learning_rate=0.01, max_depth=2, max_features=0.2, min_samples_leaf=1, min_samples_split=18, n_estimators=100, subsample=0.3)
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6933333333333334
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=1, max_features=0.4, min_samples_leaf=12, min_samples_split=11, n_estimators=100, subsample=0.9000000000000001)),
    StackingEstimator(estimator=BernoulliNB(alpha=0.1,True)),
    Binarizer(threshold=0.1),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=2, max_features=0.2, min_samples_leaf=1, min_samples_split=18, n_estimators=100, subsample=0.3)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)