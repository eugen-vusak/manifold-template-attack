from pathlib import Path

import numpy as np
import pandas as pd
# import umap
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from utils.aes import LeakageModel, get_aes_output_for_leakage
from utils.data import load_data
from utils.feature_selection import SumOfDifferenceFeatureSelector
from utils.measure import guessing_entropy_and_success_rate
from utils.sklearn_wrappers import (ColumnTransformer,
                                    TransformedTargetTransformer)
from utils.TA import TemplateAttack

DATA_ROOT = Path("data")
DATASET = "chipwhisperer"  # ascad_fixed, ascad_variable, ches_ctf, chipwhisperer
TARGET_BYTE = 0

NUM_OF_FEATURES = 50
FEATURE_SPACING = 1

GE_NUMBER_OF_EXPERIMENTS = 100
GE_NUMBER_OF_TRACES = 10

LEAKAGE_MODEL = LeakageModel.HW  # intermediate, HW

########################################################################
############################## Load data ###############################
########################################################################

train, test = load_data(DATA_ROOT/DATASET, TARGET_BYTE)

(tracesTrain, ptTrain, keyTrain) = train
(tracesTest, ptTest, keyTest) = test

X_train = np.hstack((tracesTrain, ptTrain.reshape(-1, 1)))
y_train = keyTrain
X_test = np.hstack((tracesTest, ptTest.reshape(-1, 1)))
y_test = keyTest

########################################################################
############################## Profiling ###############################
########################################################################

output_fn = get_aes_output_for_leakage(LEAKAGE_MODEL)

# feature selection
feature_sel = SelectKBest(f_regression, k=300)

# dimensionality reduction (this is what we change)
dim_rdc = SumOfDifferenceFeatureSelector(
    n_components=NUM_OF_FEATURES,
    feature_spacing=FEATURE_SPACING
)

ta = TemplateAttack(output_fn)

fs = ColumnTransformer(feature_sel, X_column=-1)
dr = ColumnTransformer(dim_rdc, X_column=-1)


def bla(X, y):
    a = output_fn(X[:, -1].astype(np.int32), y)
    return a


fs = TransformedTargetTransformer(fs, func=bla)
dr = TransformedTargetTransformer(dr, func=bla)


pipe = Pipeline([
    ("feature_sel", fs),
    ("dim_rdc",  dr),
    ("templateAttack", ta)
])

pipe.fit(X_train, y_train)
scores = pipe.predict_proba(X_test)

# search = GridSearchCV(
#     pipe,
#     param_grid={"dim_rdc__feature_spacing": [1]},
#     cv=5,
#     verbose=3,
#     scoring=ge_scoring
# )

# a = search.fit(X_train, y_train)
# scores = search.predict_proba(X_test)


# compute measures (GE and SR)
np.random.seed(43)
ge, sr = guessing_entropy_and_success_rate(
    scores,
    y_train[0],
    number_of_traces=GE_NUMBER_OF_TRACES,
    number_of_experiments=GE_NUMBER_OF_EXPERIMENTS)


print(ge)  # should tend to zero
print(sr)  # should tend to one
