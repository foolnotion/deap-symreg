import operator
import math
import random
import warnings # suppress some warnings related to invalid values
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import multiprocessing
import timeit
import time
import json

import argparse
import itertools

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from algorithms import eaElite 
from multiprocessing import Pool, Lock 

import cProfile

#set_start_method("spawn")

# parse some arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Data path (can be either a directory or a .csv file', type=str)

args = parser.parse_args()

data_path = args.data
base_path = os.path.dirname(data_path)
files = list(os.listdir(data_path)) if os.path.isdir(data_path) else [ os.path.basename(data_path) ]
files = list([f for f in files if f.endswith('.json')])

lock = Lock()

columns = [ 'Problem', 'Index', 'Elapsed', 'R2 train', 'R2 test', 'RMSE train', 'RMSE test', 'NMSE train', 'NMSE test' ]


def limit_range(values):
    try: 
        min_ = values[np.isfinite(values)].min() 
        max_ = values[np.isfinite(values)].max() 
    except ValueError:
        return np.repeat(0., len(values))

    mid_ = (min_ + max_) / 2.
    np.nan_to_num(values, copy=False, nan=mid_, posinf=mid_, neginf=mid_)

    np.clip(values, min_, max_, out=values)
    return values


def r2stat(y_train, y_pred):
    try:
        r = pearsonr(y_train, y_pred)[0]
        fit = r * r 
    except ValueError:
        fit = 0.

    if ~np.isfinite(fit):
        fit = 0.

    fit = max(min(fit, 1.), 0.) 
    return fit


def run(path, idx, results):
    with open(path, 'r') as h:
        info     = json.load(h)
        dir_name = os.path.dirname(path)
        csv_path = info['metadata']['filename']
        training = info['metadata']['training_rows']
        test     = info['metadata']['test_rows']
        target   = info['metadata']['target']

    # load data
    df = pd.read_csv(os.path.join(dir_name, csv_path), sep=',')

    X = df.loc[:, df.columns != target].to_numpy()
    y = df[target].to_numpy()

    X_train = X[training['start']:training['end']]
    X_test = X[test['start']:test['end']]

    y_train = y[training['start']:training['end']]
    y_test = y[test['start']:test['end']]

    rows, cols = X_train.shape

    prefix = path+str(idx)

    # global primitive set
    # set static height limit for all generated trees
    pset = gp.PrimitiveSet("MAIN", cols)
    pset.addPrimitive(np.add, 2, name="vadd")
    pset.addPrimitive(np.subtract, 2, name="vsub")
    pset.addPrimitive(np.multiply, 2, name="vmul")
    pset.addPrimitive(np.divide, 2, name="vdiv")
    #pset.addPrimitive(np.negative, 1, name="vneg")
    pset.addPrimitive(np.cos, 1, name="vcos")
    pset.addPrimitive(np.sin, 1, name="vsin")
    pset.addPrimitive(np.exp, 1, name="vexp")
    pset.addPrimitive(np.log, 1, name="vlog")
    pset.addEphemeralConstant("rand101_" + prefix , lambda: np.random.uniform(-1.0, 1.0)) #may be unable to pickle...


    def evaluate(individual):
        # Transform the tree expression in a callable function
        func = gp.compile(expr=individual, pset=pset)
        
        with warnings.catch_warnings(): # comment out when debugging
            warnings.simplefilter("ignore") # comment out when debugging
            
            y_pred = func(*X_train.T)

            if np.isscalar(y_pred):
                return 0.,
            
            y_pred = limit_range(y_pred)
            fit = r2stat(y_train, y_pred)
            return fit,

    np.seterr(all='ignore')

    with warnings.catch_warnings(): # comment out when debugging
        warnings.simplefilter("ignore") # comment out when debugging
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    maxHeight = 10
    maxLength = 50
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=5)
    
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register('mutEphemeral', gp.mutEphemeral, mode='one')
    toolbox.register('mutNodeReplacement', gp.mutNodeReplacement, pset=pset)
    
    mutOperators = [ toolbox.mutEphemeral, toolbox.mutNodeReplacement ]
    
    def mutOperator(*args, **kwargs):
        mut = np.random.choice(mutOperators)
        return mut(*args, **kwargs)
                   
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register('mutate', mutOperator)
    
    toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), max_value=maxHeight)) 
    toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=maxHeight)) 
    toolbox.decorate("mate", gp.staticLimit(len, maxLength))
    toolbox.decorate("mutate", gp.staticLimit(len, maxLength)) 
    
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    stats_fit  = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean) 
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    t0 = time.time()
    eaElite(pop, toolbox, cxpb=1, mutpb=0.25, ngen=1000, nelite=1, stats=mstats, halloffame=hof, verbose=False)

    best         = hof[0]
    func         = gp.compile(expr=best, pset=pset)

    y_pred_train = func(*X_train.T)
    y_pred_test  = func(*X_test.T)

    y_pred_train = limit_range(y_pred_train)
    y_pred_test  = limit_range(y_pred_test)

    # use a linear regressor to perform linear scaling of the predicted values 
    lr           = LinearRegression()
    lr.fit(y_pred_train.reshape(-1, 1), y_train.reshape(-1, 1))

    y_pred_train = np.squeeze(lr.predict(y_pred_train.reshape(-1, 1)))
    y_pred_test  = np.squeeze(lr.predict(y_pred_test.reshape(-1, 1)))

    r2_train     = r2stat(y_train, y_pred_train)
    r2_test      = r2stat(y_test, y_pred_test)

    mse_train    = mean_squared_error(y_train, y_pred_train)
    mse_test     = mean_squared_error(y_test, y_pred_test)

    rmse_train   = np.sqrt(mse_train)
    rmse_test    = np.sqrt(mse_test)

    nmse_train   = mse_train / np.var(y_train)
    nmse_test    = mse_test / np.var(y_test)

    t1 = time.time()

    problem = os.path.basename(path).replace('.json', '')

    with lock:
        results.append(pd.DataFrame([[problem, idx+1, t1-t0, r2_train, r2_test, rmse_train, rmse_test, nmse_train, nmse_test]], columns=columns))


if __name__ == "__main__":
    reps = 50 
    names = list([ os.path.join(base_path, f) for f in files])

    df_elapsed = pd.DataFrame(columns=['Problem', 'Reps', 'Elapsed'])
    manager = multiprocessing.Manager()
    results = manager.list()

    for name in names:
        t0 = time.time()
        pool = Pool()
        pool.starmap(run, itertools.product([ name ], np.arange(0, reps), [ results ]))
        t1 = time.time()
        print('{}\t{}\t{}'.format(name, reps, t1-t0))
        df_elapsed.loc[len(df_elapsed)] = [ os.path.basename(name).replace('.json', ''), reps, t1-t0 ]

    df_results = pd.concat(results).reset_index(drop=True)
    df_results.to_csv('deap_results.csv', index=False)
    df_elapsed.to_csv('deap_elapsed.csv', index=False)
