from pathlib import Path

# import umap
from sklearn.feature_selection import SelectKBest, f_regression

from utils.aes import LeakageModel, get_aes_output_for_leakage
from utils.data import load_data
from utils.feature_selection import SumOfDifferenceFeatureSelector
from utils.measure import guessing_entropy_and_success_rate
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

########################################################################
############################## Profiling ###############################
########################################################################

# feature selection
feature_sel = SelectKBest(f_regression, k=300)

# dimensionality reduction (this is what we change)
dim_rdc = SumOfDifferenceFeatureSelector(
    n_components=NUM_OF_FEATURES,
    feature_spacing=FEATURE_SPACING
)

# # This is example how to change dimensionality reduction that is used
# dim_rdc = umap.UMAP(n_neighbors=40, n_components=50)

output_fn = get_aes_output_for_leakage(LEAKAGE_MODEL)

ta = TemplateAttack(output_fn)

# calculate hamming value of sbox output
# for the first bit of the plain text
outputTrain = output_fn(ptTrain, keyTrain)

# feature selection fit_transform on train data
tracesTrain = feature_sel.fit_transform(tracesTrain, outputTrain)

# dimensionality reduction fit_transform on train data
tracesTrain = dim_rdc.fit_transform(tracesTrain, outputTrain)

# create templateci
ta.create_template(tracesTrain, output=outputTrain)

########################################################################
################################ Attack ################################
########################################################################

# feature selection transform test data
tracesTest = feature_sel.transform(tracesTest)

# dimensionality reduction transform test data
tracesTest = dim_rdc.transform(tracesTest)

# get probability densities for every possible keyguess
ta_logpdfs = ta.logpdfs(tracesTest, ptTest)

# compute measures (GE and SR)
ge, sr = guessing_entropy_and_success_rate(
    ta_logpdfs, keyTest[0],
    number_of_traces=GE_NUMBER_OF_TRACES,
    number_of_experiments=GE_NUMBER_OF_EXPERIMENTS)


print(ge)  # should tend to zero
print(sr)  # should tend to one
