from utils.TA import TemplateAttack
from utils.measure import guessing_entropy_and_success_rate


def run_experiment(data, output_fn, dim_rdc, n_traces, n_experiments):

    # unpack data
    ((tracesTrain, ptTrain, keyTrain),
     (tracesTest, ptTest, keyTest)) = data

    ########################################################################
    ############################## Profiling ###############################
    ########################################################################

    ta = TemplateAttack(output_fn)

    # calculate hamming value of sbox output
    # for the first bit of the plain text
    outputTrain = output_fn(ptTrain, keyTrain)

    # dimensionality reduction fit_transform on train data
    tracesTrain = dim_rdc.fit_transform(tracesTrain, outputTrain)

    # create template
    ta.create_template(tracesTrain, output=outputTrain)

    ########################################################################
    ################################ Attack ################################
    ########################################################################

    # dimensionality reduction transform test data
    tracesTest = dim_rdc.transform(tracesTest)

    # get probability densities for every possible keyguess
    ta_logpdfs = ta.logpdfs(tracesTest, ptTest)

    # compute measures (GE and SR)
    # currently only working if all keys are same in keyTest
    ge, sr = guessing_entropy_and_success_rate(
        ta_logpdfs, keyTest[0],
        number_of_traces=n_traces,
        number_of_experiments=n_experiments)

    return ge, sr
