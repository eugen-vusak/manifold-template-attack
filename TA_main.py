from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
# import umap
from sklearn.feature_selection import (SelectKBest, VarianceThreshold,
                                       f_regression)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from utils.aes import LeakageModel, get_aes_output_for_leakage
from utils.data import load_data
from utils.feature_selection import SumOfDifferenceFeatureSelector
from utils.measure import guessing_entropy_and_success_rate
from utils.sklearn_wrappers import TransformedTargetTransformer
from utils.TA import TemplateAttack

DATA_ROOT = Path("data")
DATASET = "chipwhisperer"  # ascad_fixed, ascad_variable, ches_ctf, chipwhisperer
TARGET_BYTE = 0

NUM_OF_FEATURES = 100
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

# X = (traces | plain_text)
# y = key
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


transformerPipe = Pipeline([
    ("var", VarianceThreshold()),
    ("feature_sel", feature_sel),
    ("dim_rdc", dim_rdc)
])


tracesTransformer = ColumnTransformer(
    transformers=[("ct", transformerPipe, slice(-1))],
    remainder="passthrough"
)


def transform_key_to_output(X, y):
    plain = X[:, -1].astype(np.int8)
    key = y
    return output_fn(plain, key)


tracesTransformer = TransformedTargetTransformer(
    tracesTransformer,
    func=transform_key_to_output
)

pipe = Pipeline([
    ("tracesTransformer", tracesTransformer),
    ("templateAttack", ta)
])

# pipe.fit(X_train, y_train)
# scores = pipe.predict_proba(X_test)


def ge_scoring(clf, X, y, number_of_experiments):
    scores = clf.predict_proba(X)

    # compute measures (GE and SR)
    np.random.seed(43)
    ge, sr = guessing_entropy_and_success_rate(
        scores,
        y[0],
        number_of_experiments=number_of_experiments)

    return ge[-1]


search = GridSearchCV(
    pipe,
    param_grid={
        "tracesTransformer__ct__dim_rdc__feature_spacing": [1, 2, 3, 4]
    },
    cv=5,
    verbose=3,
    scoring=ge_scoring

)

a = search.fit(X_train, y_train)
scores = search.predict_proba(X_test)


# compute measures (GE and SR)
np.random.seed(43)
ge, sr = guessing_entropy_and_success_rate(
    scores,
    y_train[0],
    number_of_experiments=GE_NUMBER_OF_EXPERIMENTS
)


print(ge)  # should tend to zero
print(sr)  # should tend to one
