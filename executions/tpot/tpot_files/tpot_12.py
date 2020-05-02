# Best pipeline: LinearSVC(SGDClassifier(Binarizer(StandardScaler(input_matrix), threshold=0.15000000000000002), alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=0.25, learning_rate=invscaling, loss=log, penalty=elasticnet, power_t=0.0), C=0.1, dual=False, loss=squared_hinge, penalty=l2, tol=0.1)
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6774117647058825
exported_pipeline = make_pipeline(
    StandardScaler(),
    Binarizer(threshold=0.15000000000000002),
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=0.25, learning_rate="invscaling", loss="log", penalty="elasticnet", power_t=0.0)),
    LinearSVC(C=0.1, dual=False, loss="squared_hinge", penalty="l2", tol=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)