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

# parse some arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Data path (can be either a directory or a .csv file', type=str)

args = parser.parse_args()

data_path = args.data

with open(data_path, 'r') as h:
    info     = json.load(h)
    dir_name = os.path.dirname(data_path)
    csv_path = info['metadata']['filename']
    training = info['metadata']['training_rows']
    test     = info['metadata']['test_rows']
    target   = info['metadata']['target']

print('target = ', target)
# load data
df = pd.read_csv(os.path.join(dir_name, csv_path), sep=',')

X = df.loc[:, df.columns != target].to_numpy()
y = df[target].to_numpy()

X_train = X[training['start']:training['end']]
X_test = X[test['start']:test['end']]

y_train = y[training['start']:training['end']]
y_test = y[test['start']:test['end']]

rows, cols = X_train.shape

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
pset.addEphemeralConstant("rand101", lambda: np.random.uniform(-1.0, 1.0)) #may be unable to pickle...


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
 
    
def evolve():
    np.seterr(all='ignore')
    random.seed(318)
        
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    maxHeight = 10
    maxLength = 50
    
    toolbox = base.Toolbox()
    from multiprocessing.pool import Pool
    pool = Pool()
    toolbox.register("map", pool.map)
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=5)
    
    # Allow for random choice between 2 set up mutators
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register('mutUniform', gp.mutUniform,   expr=toolbox.expr_mut, pset=pset)
    toolbox.register('mutEphemeral', gp.mutEphemeral, mode='all')
    toolbox.register('mutNodeReplacement', gp.mutNodeReplacement, pset=pset)
    
    mutOperators = [ toolbox.mutUniform, toolbox.mutEphemeral, toolbox.mutNodeReplacement ]
    
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

    eaElite(pop, toolbox, cxpb=1, mutpb=0.25, ngen=1000, nelite=1, stats=mstats, halloffame=hof, verbose=False)

    # print("\nBest Hof:\n%s"%hof[0])

    return pop, mstats, hof


if __name__ == "__main__":
    np.random.seed(0)
    t0 = time.time()
    pop, stats, hof = evolve()
    t1 = time.time()

    best = hof[0]
    func = gp.compile(expr=best, pset=pset)

    y_pred_train = func(*X_train.T)
    y_pred_test  = func(*X_test.T)

    y_pred_train = limit_range(y_pred_train)
    y_pred_test = limit_range(y_pred_test)

    # use a linear regressor to perform linear scaling of the predicted values 
    lr = LinearRegression()
    lr.fit(y_pred_train.reshape(-1, 1), y_train.reshape(-1, 1))

    y_pred_train = np.squeeze(lr.predict(y_pred_train.reshape(-1, 1)))
    y_pred_test  = np.squeeze(lr.predict(y_pred_test.reshape(-1, 1)))

    print(y_pred_train.shape)


    r2_train     = r2stat(y_train, y_pred_train)
    r2_test      = r2stat(y_test, y_pred_test)

    mse_train    = mean_squared_error(y_train, y_pred_train)
    mse_test     = mean_squared_error(y_test, y_pred_test)

    rmse_train   = np.sqrt(mse_train)
    rmse_test    = np.sqrt(mse_test)

    nmse_train   = mse_train / np.var(y_train)
    nmse_test    = mse_test / np.var(y_test)

    print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(t1 - t0, r2_train, r2_test, rmse_train, rmse_test, nmse_train, nmse_test))

