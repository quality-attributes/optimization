# Best pipeline: GradientBoostingClassifier(ZeroCount(ZeroCount(SelectPercentile(CombineDFs(input_matrix, input_matrix), percentile=67))), learning_rate=0.01, max_depth=6, max_features=0.3, min_samples_leaf=1, min_samples_split=19, n_estimators=100, subsample=0.9500000000000001)
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6737254901960785
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    SelectPercentile(score_func=f_classif, percentile=67),
    ZeroCount(),
    ZeroCount(),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=6, max_features=0.3, min_samples_leaf=1, min_samples_split=19, n_estimators=100, subsample=0.9500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)