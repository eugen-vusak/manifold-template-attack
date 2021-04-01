import os
from datetime import datetime
from math import floor
from pathlib import Path

import umap  # takes a bit to import
from sklearn import manifold
from sklearn.decomposition import PCA

from experiment import run_experiment
from utils.aes import LeakageModel, get_aes_output_for_leakage
from utils.data import load_data
from utils.feature_selection import SumOfDifferenceFeatureSelector

PREVIEW = False

DATA_ROOT = Path("data")
TARGET_BYTE = 0

GE_N_EXPERIMENTS = 100


datasets = [
    #"chipwhisperer",
    "ascad_fixed",
    "ascad_variable",
    "ches_ctf"
]

leakage_models = [
    #LeakageModel.intermediate,
    LeakageModel.HW
]


######################## manifold transform parameters ########################
# for MLLE: n_neighbors >= n_components
# for HLLE: n_neighbors > n_components * (n_components + 3) / 2
mf_parameters_dict = {
    "n_components": {
        10: {"n_neighbors": [20, 50, 200]},
        25: {"n_neighbors": [50, 125, 500]},
        50: {"n_neighbors": [100, 250, 1000]},
        75: {"n_neighbors": [150, 375, 1500]},
        100: {"n_neighbors": [200, 500, 2000]},
    },
}

mf_n_components = mf_parameters_dict["n_components"].keys()
mf_n_jobs = None

############################ SumOfDifference parameters #######################
sod_n_components = mf_n_components
sod_feature_spacing = 5


################################ PCA parameters ###############################
pca_n_components = mf_n_components

################################ end parameters ###############################


def generate_manifold_tranformations(n_components):

    isomap_clf = manifold.Isomap(
        n_components=n_components,
        n_jobs=mf_n_jobs
    )

    lle_clf = manifold.LocallyLinearEmbedding(
        n_components=n_components,
        method='standard',
        n_jobs=mf_n_jobs
    )

    mlle_clf = manifold.LocallyLinearEmbedding(
        n_components=n_components,
        method='modified',
        n_jobs=mf_n_jobs
    )

    # hlle_clf = manifold.LocallyLinearEmbedding(
    #     n_components=n_components,
    #     method='hessian',
    #     n_jobs=mf_n_jobs
    # )

    ltsa_clf = manifold.LocallyLinearEmbedding(
        n_components=n_components,
        method='ltsa',
        n_jobs=mf_n_jobs
    )

    umap_clf = umap.UMAP(
        n_components=n_components,
    )

    # manifold transformations
    trans_clfs = [
        (isomap_clf, "isomap"),
        (lle_clf, "lle"),
        (mlle_clf, "mlle"),
        # (hlle_clf, "hlle"),
        (ltsa_clf, "ltsa"),
        (umap_clf, "umap"),
    ]

    return trans_clfs


def generate_all_dim_reductors():
    """This function generates all dimensionality reduction algorithms
    that are used for all experiments. Dimensionality reductor is a tuple 
    consisting of transformer object and its name. 

    Yields:
        (transformer, str, dict): a tuple with transformer, its short name and param_dict
    """

    # # SumOfDifference dimensionality reductors (TA default reductor)
    for n_components in sod_n_components:

        sod_dim_rdc = SumOfDifferenceFeatureSelector(
            n_components=n_components,
            feature_spacing=min(floor(100 / n_components), 5)
        )
        yield (sod_dim_rdc, f"sod_{n_components}", None)

    # # PCA (probably good idea to test it as well)
    for n_components in pca_n_components:
        pca_dim_rdc = PCA(n_components=n_components)
        yield (pca_dim_rdc, f"pca_{n_components}", None)

    # all manifold reductors
    for n_components in mf_n_components:

        manifold_dim_rds = generate_manifold_tranformations(
            n_components=n_components
        )

        param_dict = mf_parameters_dict["n_components"][n_components]
        for dim_rdc, dim_rdc_name in manifold_dim_rds:
            yield (dim_rdc,  f"{dim_rdc_name}_{n_components}", param_dict)


################################### main ######################################
# create output dir
now = datetime.now()
timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
folderName = "run_" + timestamp
directory = folderName
parent_dir = "results/"
path = os.path.join(parent_dir, directory)
os.makedirs(path)




for dataset in datasets:
    data = load_data(DATA_ROOT/dataset, TARGET_BYTE)

    for leakage_model in leakage_models:
        # aes_output_fn is used to generate output of produced by aes sbox operation
        # if traces is X, aes_output_fn(plain, key) can be seen as y
        aes_output_fn = get_aes_output_for_leakage(leakage_model)

        for dim_rdc, dim_rdc_name, param_dict in generate_all_dim_reductors():

            print("Running experiment:", end=" ")
            print(dataset, leakage_model.name, dim_rdc_name, sep=" - ")
            
            now = datetime.now()
            timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
            trainSize= data[0]
            testSize = data[1]
            logName = "log.txt"
            logFile = open("results/"+folderName+"/"+logName, "a+")
            logFile.write("********* " + timestamp + "\n")
            logFile.write("Running experiment: " + dataset + " - " + leakage_model.name + " - " + dim_rdc_name  )
            logFile.write("\n")
            logFile.write("Dataset size: train - " + str(len(trainSize[1])) + ", test - " + str(len(testSize[1])))
            logFile.write("\n")

            if PREVIEW:
                continue

            # try to run experimnt
            ge = sr = gridSearch_res = fail_msg = None
            
            try:
                ge, sr, gridSearch_res = run_experiment(
                    data, aes_output_fn,
                    dim_rdc, param_dict,
                    GE_N_EXPERIMENTS
                )
            except Exception as e:
                 print("Experiment failed:", e)
                 fail_msg = str(e)

            # report to file
            # create file
            fileName = dataset + "_" + leakage_model.name + "_" + dim_rdc_name + ".txt"
            resFile = open("results/"+folderName+"/"+fileName, "w+")
            
            # write to file
            if fail_msg:
                resFile.write("!!!Experiment failed: " + fail_msg)
                logFile.write("!!!Experiment failed: " + fail_msg)
                logFile.write("\n")
            else:
                if gridSearch_res != None:
                    resFile.write("Grid Search params:")
                    resFile.write("\n")
                    resFile.write("best_params_: %s;" % gridSearch_res.best_params_)  
                    resFile.write("cv_results_: %s;" % gridSearch_res.cv_results_) 
                    resFile.write("\n")
                if gridSearch_res == None:
                    resFile.write("No Grid Search.\n")
                if ge.size > 0:
                    resFile.write("ge:")
                    resFile.write("\n")
                    for item in ge:
                        resFile.write("%s;" % item)
                    resFile.write("\n")
                if sr.size > 0:
                    resFile.write("sr:")
                    resFile.write("\n")
                    for item in sr:
                        resFile.write("%s;" % item)
                    resFile.write("\n")

            resFile.close()
            logFile.write("\n\n")
            logFile.close()
