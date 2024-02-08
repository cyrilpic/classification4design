"""Code for classification study.

This files contains the code the train several classifiers using 
Bayesian optimization to search for optimal hyperparameters. The code
is adapted from the code from Lyle Regenwetter, but translated to SMAC.
"""

import time
from typing import Union

import numpy as np
import sklearn
from sklearn.utils.parallel import delayed, Parallel
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from ConfigSpace import Configuration
from tqdm import tqdm

from .classifiers.utils import ClassifierModule


def train(
    model_module: ClassifierModule,
    X: np.ndarray,
    y: np.ndarray,
    config: Configuration,
    seed: int = 0,
) -> float:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    parallel = Parallel(n_jobs=None, pre_dispatch="2*n_jobs")
    scores = parallel(
        delayed(model_module.fit_score)(config, X, y, train, test, seed)
        for train, test in cv.split(X, y)
    )
    return 1 - np.mean(scores)


# """
# Outermost Function Call for hyperparameter tuning.
# Takes in dataframe of optimization parameters, datasets, parameter names, algorithm name
# Takes in reduction parameter for Generalized Subspace Design for grid search (reduction)
# Takes in the number of retrainings for each hyperparameter config (n_trials), number of Cross Val splits (n_splits),
# number of iterations to run BO (n_bo) number of instantiations of the final hyperparameter configuration to test
# First, we set up a wrapper function for the call to fit_bo that we can use with Optuna.
# Nest, we set up the pruner and the study, then perform the initial grid sampling.
# We then reset the sampler for our study to the TPESampler and perform BO to find the best hyperparameter configuration
# Finally, we call preparDF, saving a report, and search for an optimal instantiation of a model to return.
# """


# def hyperparam_search(
#     df,
#     function,
#     x_train,
#     x_test,
#     y_train,
#     y_test,
#     name,
#     reduction=1,
#     n_trials=10,
#     n_splits=5,
#     n_bo=100,
#     n_inst=100,
#     sklearn=True,
# ):
#     def wrapper(trial):  # Wrapper function for fit_bo
#         return fit_bo(trial, function, df, x_train, y_train, n_trials, n_splits)

#     start_time = time.perf_counter()
#     pruner = optuna.pruners.PercentilePruner(
#         75.0, n_startup_trials=1, n_warmup_steps=3, interval_steps=1
#     )  # Pruner
#     study = optuna.create_study(
#         direction="maximize", pruner=pruner
#     )  # Create a new study using pruner.
#     init_gridsample(
#         study, wrapper, df, reduction=reduction
#     )  # Perform initial grid sampling
#     study.sampler = (
#         optuna.samplers.TPESampler()
#     )  # Set sampler to Tree-Structured Parzen Estimator
#     study.optimize(wrapper, n_trials=n_bo)  # Optimize
#     bestparams = study.best_params
#     model = test_instantiations(function, bestparams, x_train, y_train, n_inst)
#     train_time = time.perf_counter() - start_time
#     model._total_train_time = train_time
#     # bestf1, auc, precision, recall, accuracy = score(model, x_test, y_test, sklearn)
#     # save_results(name, n_configs, reduction, n_trials, n_splits, n_bo, n_inst, start_time, bestf1, auc, precision, recall, accuracy)
#     return model


# """
# Perform the initialization of the BO using grid sampling.
# First call get_configlist to get the list of grid sampling configurations
# Then for each configuration, initialize a single point grid sample search in optuna
# Run 1 trial for each grid point
# """


# def init_gridsample(study, func, df, reduction=5):
#     configs = get_configlist(df, reduction)  # Get the DOE configurations
#     for config in configs:  # Loop over all DOE configs
#         # Create the search space
#         search_space = {}
#         for i in range(len(df.index)):
#             search_space[df.index[i]] = [config[i]]

#         # Set the study's sampler to be single point gridsearch space, then do one round of fitting
#         study.sampler = optuna.samplers.GridSampler(search_space)
#         study.optimize(func, n_trials=1)
#     return len(configs)


# """
# Calculate gridsearch configurations. If we have a reduction greater than one, use GSD to reduce
# For each configuration, calculate the corresponding parameter values using get_gridvalm then return
# """


# def get_configlist(df, reduction=5):
#     # Create a list of the number of grid locations for parameter.
#     # Cont & int get 2 locations, Cat gets locations equal to # of discrete categories
#     configsize = []
#     for parameter in df.index:
#         if df.loc[parameter, "Datatype"] == "Categorical":
#             configsize.append(len(df.loc[parameter, "Values/Min-Max"]))
#         else:
#             configsize.append(df.loc[parameter, "Gridres"])

#     print(configsize)

#     # If reduction>1 we call gsd, otherwise do full fact
#     if reduction > 1:
#         DOEconfigs = pyDOE2.gsd(configsize, reduction=reduction)
#     else:
#         DOEconfigs = pyDOE2.fullfact(configsize).astype(int)
#     print("Number of GridSearch configs: " + str(len(DOEconfigs)))

#     # Look up actual gridsearch values from indices
#     configvals = []
#     for i in range(len(DOEconfigs)):
#         newvals = []
#         for j in range(len(DOEconfigs[0])):
#             newvals.append(get_gridval(df, df.index[j], DOEconfigs[i][j]))
#         configvals.append(newvals)
#     return configvals


# """
# Calculate the value of a particular grid sampling point
# Categorical, continuous, and discrete variables are handled individually
# Log scale our calculations when the variable is log scaled
# We assume grid points are evenly spaced between limits
# """


# def get_gridval(df, parameter, index):
#     # If categorical, we simply return the value corresponidng to the index
#     if df.loc[parameter, "Datatype"] == "Categorical":
#         return df.loc[parameter, "Values/Min-Max"][index]

#     # Grab vals for minval, maxval and scaling from df for convenience
#     minval = df.loc[parameter, "Values/Min-Max"][0]
#     maxval = df.loc[parameter, "Values/Min-Max"][1]
#     scaling = df.loc[parameter, "Logscaling"]
#     gridres = df.loc[parameter, "Gridres"]

#     # Calculate the percentile between the parameter limits to sample the grid point
#     gridloc = index / gridres + 1 / gridres / 2

#     # scaling is true, we will logscale when performing our calculations
#     if scaling is True:
#         # Indices 0 and 1 should be at 25th and 75th percentile of parameter ranges, respecively
#         value = np.exp((1 - gridloc) * np.log(minval) + gridloc * np.log(maxval))
#     else:
#         value = (1 - gridloc) * minval + gridloc * maxval

#     # If we have an integer parameter, we round to the nearest integer
#     if df.loc[parameter, "Datatype"] == "Continuous":
#         return value
#     if df.loc[parameter, "Datatype"] == "Integer":
#         return round(value)


# """
# Sample a particular parameter from the optuna trial.
# Automates the sampling call based on the information about the parameter contained in DF

# This function duals as a simple dictionary lookup if trial is a dictionary.
# This allows the reuse of fit_XXX functions when selecting an instantiation for a given config
# """


# def get_params(trial, df):
#     # If we have passed in a dictionary, simply index a value from the dictionary.
#     if type(trial) == dict:
#         return trial

#     # Otherwise, we are using optuna. Loop over all parameters in the DF
#     params = {}
#     for parameter in df.index:
#         # We setup each call to the trial in the format Optuna expects. See the Optuna docs
#         if df.loc[parameter, "Datatype"] == "Categorical":
#             params[parameter] = trial.suggest_categorical(
#                 parameter, df.loc[parameter, "Values/Min-Max"]
#             )
#         else:
#             # Grab vals for minval, maxval and scaling from df for convenience
#             minval = df.loc[parameter, "Values/Min-Max"][0]
#             maxval = df.loc[parameter, "Values/Min-Max"][1]
#             scaling = df.loc[parameter, "Logscaling"]
#             if df.loc[parameter, "Datatype"] == "Continuous":
#                 params[parameter] = trial.suggest_float(
#                     parameter, minval, maxval, log=scaling
#                 )
#             elif df.loc[parameter, "Datatype"] == "Integer":
#                 params[parameter] = trial.suggest_int(
#                     parameter, minval, maxval, 1, log=scaling
#                 )
#     return params


# """
# General BO fit loop. For each trial in BO we will create n_trials k_fold splits where k is n_splits
# This yields a total of n_trial*n_folds fitting runs.
# Since we apply the hyperparameters to each model differently, we call the func function, which is specified
# This func function will be a unique function for each type of model which will assign the hyperparameters
# The func function will then fit the model and return the model back. For each run, we score on the val set.
# We pass intermediate scores in a report to Optuna so it can determine if the trial should be pruned
# If the trial is pruned, the trial prematurely exits
# """


# def fit_bo(trial, func, df, xdata, ydata, n_trials, n_splits):
#     valf1 = 0
#     stepcount = 0
#     params = get_params(trial, df)  # Get the parameters for this trial
#     with tqdm(total=n_splits * n_trials) as pbar:  # Progress bar
#         for j in range(n_trials):
#             kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
#             for train_index, test_index in kf.split(
#                 xdata
#             ):  # for each trial, loop over k-fold split
#                 x_train, x_val = xdata[train_index], xdata[test_index]
#                 y_train, y_val = ydata[train_index], ydata[test_index]
#                 model = func(
#                     params, x_train, x_val, y_train, y_val
#                 )  # Call the specified model init+fit function
#                 instancef1 = sklearn.metrics.f1_score(
#                     y_val, np.rint(model.predict(x_val))
#                 )
#                 valf1 += instancef1
#                 trial.report(
#                     valf1 / (stepcount + 1), stepcount
#                 )  # Report scores for potential early pruning
#                 if trial.should_prune():  # Exit if Optuna decides to prune
#                     raise optuna.exceptions.TrialPruned()
#                 pbar.update(1)  # Update progress bar
#                 stepcount += 1
#         return valf1 / n_splits / n_trials


# """
# Select an optimal instantiation of a model.
# We have defined a constant random seed to ensure train and val sets across all models are consistent
# numtest models are tested, all instantiated using bestparams.
# """


# def test_instantiations(func, bestparams, xdata, ydata, numtest=100, sklearn_like=True):
#     x_train, x_val, y_train, y_val = train_test_split(
#         xdata, ydata, test_size=0.2, random_state=1
#     )
#     bestf1 = float("-inf")
#     for _ in tqdm(range(numtest)):
#         # True flag indicates to we are testing instantiations. Used to predict probability in models like SVM
#         # We need probabilities for AUC, but they cause slower fitting
#         model = func(bestparams, x_train, x_val, y_train, y_val, True)
#         valf1 = sklearn.metrics.f1_score(y_val, np.rint(model.predict(x_val)))
#         if valf1 > bestf1:
#             bestf1 = valf1
#             bestmodel = model

#     return bestmodel
