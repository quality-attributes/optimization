# Best pipeline: ExtraTreesClassifier(KNeighborsClassifier(Normalizer(KNeighborsClassifier(CombineDFs(CombineDFs(CombineDFs(input_matrix, KNeighborsClassifier(input_matrix, n_neighbors=24, p=2, weights=uniform)), input_matrix), SelectPercentile(Normalizer(CombineDFs(input_matrix, input_matrix), norm=l2), percentile=67)), n_neighbors=24, p=2, weights=uniform), norm=max), n_neighbors=24, p=2, weights=uniform), bootstrap=False, criterion=gini, max_features=0.7000000000000001, min_samples_leaf=7, min_samples_split=18, n_estimators=100)
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6816470588235294
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            make_union(
                FunctionTransformer(copy),
                StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=24, p=2, weights="uniform"))
            ),
            FunctionTransformer(copy)
        ),
        make_pipeline(
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            ),
            Normalizer(norm="l2"),
            SelectPercentile(score_func=f_classif, percentile=67)
        )
    ),
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=24, p=2, weights="uniform")),
    Normalizer(norm="max"),
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=24, p=2, weights="uniform")),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.7000000000000001, min_samples_leaf=7, min_samples_split=18, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)