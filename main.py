from pathlib import Path

# import umap  # takes a bit to import
from sklearn import manifold
from sklearn.decomposition import PCA

from experiment import run_experiment
from utils.aes import LeakageModel, get_aes_output_for_leakage
from utils.data import load_data
from utils.feature_selection import SumOfDifferenceFeatureSelector
from utils.parameters import generate_params_grid

PREVIEW = True

DATA_ROOT = Path("data")
TARGET_BYTE = 0

GE_N_EXPERIMENTS = 50
GE_N_TRACES = 10


datasets = [
    "chipwhisperer",
    "ascad_fixed",
    "ascad_variable",
    "ches_ctf"
]

leakage_models = [
    LeakageModel.intermediate,
    LeakageModel.HW
]

######################## manifold transform parameters ########################
# for HLLE: n_neighbors > n_components * (n_components + 3) / 2
mf_parameters_dict = {
    "n_components": {
        50: {"n_neighbors": [100, 250, 1000]},
        100: {"n_neighbors": [200, 500, 2000]},
        150: {"n_neighbors": [500, 750, 3000]},
    },
}
mf_n_jobs = None

############################ SumOfDifference parameters #######################
sod_n_components = mf_parameters_dict["n_components"].keys()
sod_feature_spacing = 5


################################ PCA parameters ###############################
pca_n_components = sod_n_components

################################ end parameters ###############################


def generate_manifold_tranformations(n_neighbors, n_components):

    name_format = f"{{}}_{n_neighbors}_{n_components}"

    isomap_clf = manifold.Isomap(
        n_neighbors=n_neighbors, n_components=n_components,
        n_jobs=mf_n_jobs
    )

    lle_clf = manifold.LocallyLinearEmbedding(
        n_neighbors=n_neighbors,  n_components=n_components,
        method='standard',
        n_jobs=mf_n_jobs
    )

    mlle_clf = manifold.LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=n_components,
        method='modified',
        n_jobs=mf_n_jobs
    )

    hlle_clf = manifold.LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=n_components,
        method='hessian',
        n_jobs=mf_n_jobs
    )

    ltsa_clf = manifold.LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=n_components,
        method='ltsa',
        n_jobs=mf_n_jobs
    )

    # umap_clf = umap.UMAP(
    #     n_neighbors=n_neighbors, n_components=n_components
    # )

    # manifold transformations
    trans_clfs = [
        (isomap_clf, name_format.format("isomap")),
        (lle_clf, name_format.format("lle")),
        (mlle_clf, name_format.format("mlle")),
        (hlle_clf, name_format.format("hlle")),
        (ltsa_clf, name_format.format("ltsa")),
        # (umap_clf, name_format.format("umap")),
    ]

    return trans_clfs


def generate_all_dim_reductors():
    """This function generates all dimensionality reduction algorithms
    that are used for all experiments. Dimensionality reductor is a tuple 
    consisting of transformer object and its name. 

    Yields:
        (transformer, str): a tuple with transformer and its short name
    """

    # SumOfDifference dimensionality reductors (TA default reductor)
    for n_components in sod_n_components:

        sod_dim_rdc = SumOfDifferenceFeatureSelector(
            n_components=n_components,
            feature_spacing=sod_feature_spacing
        )
        yield (sod_dim_rdc, f"sod_{n_components}")

    # PCA (probably good idea to test it as well)
    for n_components in pca_n_components:
        pca_dim_rdc = PCA(n_components=n_components)
        yield (pca_dim_rdc, f"pca_{n_components}")

    # all manifold reductors
    for params in generate_params_grid(mf_parameters_dict):

        manifold_dim_rds = generate_manifold_tranformations(
            n_components=params["n_components"],
            n_neighbors=params["n_neighbors"]
        )

        yield from manifold_dim_rds


################################### main ######################################

for dataset in datasets:
    data = load_data(DATA_ROOT/dataset, TARGET_BYTE)

    for leakage_model in leakage_models:
        # aes_output_fn is used to generate output of produced by aes sbox operation
        # if traces is X, aes_output_fn(plain, key) can be seen as y
        aes_output_fn = get_aes_output_for_leakage(leakage_model)

        for dim_rdc, dim_rdc_name in generate_all_dim_reductors():

            print("Running experiment:", end=" ")
            print(dataset, leakage_model.name, dim_rdc_name, sep=" - ")

            if PREVIEW:
                continue

            # try to run experimnt
            ge = sr = fail_msg = None
            try:
                ge, sr = run_experiment(
                    data, aes_output_fn, dim_rdc,
                    GE_N_TRACES, GE_N_EXPERIMENTS
                )
            except Exception as e:
                fail_msg = str(e)

            # print(ge)
            # print(sr)
            # print(fail_msg)

            # report to file or somewhere
