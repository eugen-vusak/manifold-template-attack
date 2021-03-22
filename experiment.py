from utils.sklearn_wrappers import TransformedTargetTransformer
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from utils.measure import guessing_entropy_and_success_rate, make_ge_scoring
from utils.TA import TemplateAttack


def run_experiment(data, output_fn, dim_rdc, dim_rdc_param_grid, n_experiments):

    # Unpack data
    ((tracesTrain, ptTrain, keyTrain),
     (tracesTest, ptTest, keyTest)) = data

    # Prepare data
    ##  X = (traces | plain_text)
    ##  y = key
    X_train = np.hstack((tracesTrain, ptTrain.reshape(-1, 1)))
    y_train = keyTrain
    X_test = np.hstack((tracesTest, ptTest.reshape(-1, 1)))
    y_test = keyTest

    # Build model
    tracesTransformer = Pipeline([
        ("var",             VarianceThreshold()),
        ("feature_sel",     SelectKBest(f_regression, k=300)),
        ("dim_rdc",         dim_rdc)
    ])

    tracesTransformer = ColumnTransformer(
        transformers=[("ct", tracesTransformer, slice(-1))],
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

    model = Pipeline([
        ("tracesTrans", tracesTransformer),
        ("TA",  TemplateAttack(output_fn))
    ])

    if dim_rdc_param_grid is not None:

        param_grid = {
            f"tracesTrans__ct__dim_rdc__{k}": v
            for k, v in dim_rdc_param_grid.items()
        }

        model = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=5,
            verbose=3,
            scoring=make_ge_scoring(n_experiments)

        )

    # Profiling
    model.fit(X_train, y_train)

    # Attack
    scores = model.predict_proba(X_test)

    # compute measures (GE and SR)
    ge, sr = guessing_entropy_and_success_rate(
        scores,
        y_train[0],
        number_of_experiments=n_experiments
    )

    return ge, sr
