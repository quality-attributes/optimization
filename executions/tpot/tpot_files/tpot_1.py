# Best pipeline: RandomForestClassifier(CombineDFs(SelectFromModel(Normalizer(BernoulliNB(input_matrix, alpha=1.0,True), norm=l2), criterion=gini, max_features=0.8, n_estimators=100, threshold=0.0), SelectPercentile(MaxAbsScaler(Binarizer(ZeroCount(input_matrix), threshold=0.0)), percentile=49)), bootstrap=False, criterion=gini, max_features=0.05, min_samples_leaf=3, min_samples_split=11, n_estimators=100)
# TPOTClassifier(config_dict=None, crossover_rate=0.1, cv=5,
#                disable_update_check=False, early_stop=None, generations=100,
#                max_eval_time_mins=5, max_time_mins=None, memory=None,
#                mutation_rate=0.9, n_jobs=1, offspring_size=None,
#                periodic_checkpoint_folder=None, population_size=100,
#                random_state=None, scoring=None, subsample=1.0, template=None,
#                use_dask=False, verbosity=2, warm_start=False)

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer, MaxAbsScaler, Normalizer
from tpot.builtins import StackingEstimator, ZeroCount

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6814901960784313
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            StackingEstimator(estimator=BernoulliNB(alpha=1.0,True)),
            Normalizer(norm="l2"),
            SelectFromModel(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.8, n_estimators=100), threshold=0.0)
        ),
        make_pipeline(
            ZeroCount(),
            Binarizer(threshold=0.0),
            MaxAbsScaler(),
            SelectPercentile(score_func=f_classif, percentile=49)
        )
    ),
    RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.05, min_samples_leaf=3, min_samples_split=11, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)