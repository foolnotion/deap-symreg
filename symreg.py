import operator
import math
import random
import warnings # suppress some warnings related to invalid values

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import multiprocessing
import timeit

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from algorithms import eaElite 

# load data
df = pd.read_csv('./data/Poly-10.csv', sep=',')
#print df
X = df.iloc[:,:-1].to_numpy()
y = df.iloc[:,-1].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234)
rows, cols = X_train.shape

print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)

# set static height limit for all generated trees
pset = gp.PrimitiveSet("MAIN", cols)
pset.addPrimitive(np.add, 2, name="vadd")
pset.addPrimitive(np.subtract, 2, name="vsub")
pset.addPrimitive(np.multiply, 2, name="vmul")
pset.addPrimitive(np.divide, 2, name="vdiv")
pset.addPrimitive(np.negative, 1, name="vneg")
pset.addPrimitive(np.cos, 1, name="vcos")
pset.addPrimitive(np.sin, 1, name="vsin")
pset.addPrimitive(np.exp, 1, name="vexp")
pset.addPrimitive(np.log, 1, name="vlog")
pset.addEphemeralConstant("rand101", lambda: np.random.uniform(-1.0, 1.0)) #may be unable to pickle...


def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = gp.compile(expr=individual, pset=pset)
    
    with warnings.catch_warnings(): # comment out when debugging
        warnings.simplefilter("ignore") # comment out when debugging
        
        y_pred = func(*X_train.T)
        
        if np.isscalar(y_pred):
            y_pred = np.repeat(y_pred, rows)
        
        min_ = np.nanmin(y_pred)
        max_ = np.nanmax(y_pred)
        
        if ~np.isfinite(min_) or ~np.isfinite(max_):
            return -1000.,
        
        mid_ = (min_ + max_) / 2
        # y_pred[np.where(~np.isfinite(y_pred))] = mid_ # using older numpy so manually doing nan_to_num
        np.nan_to_num(y_pred, copy=False, nan=mid_, posinf=mid_, neginf=mid_)

        fit = r2_score(y_train, y_pred)
        
        if ~np.isfinite(fit):
            fit = -1000.,

        fit = np.clip(fit, -1000., 1.) #expensife for a single float
        
        return fit,

    
def main():
    np.seterr(all='ignore')
    random.seed(318)
        
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    maxHeight = 10
    maxLength = 50
    
    toolbox = base.Toolbox()
    from multiprocessing.pool import Pool
    pool = Pool(24)
    toolbox.register("map", pool.map)
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=maxHeight)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("select", tools.selTournament, tournsize=5)
    
    # Allow for random choice between 2 set up mutators
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register('mutUniform', gp.mutUniform,   expr=toolbox.expr_mut, pset=pset)
    toolbox.register('mutEphemeral', gp.mutEphemeral, mode='all')
    toolbox.register('mutNodeReplacement', gp.mutNodeReplacement, pset=pset)
    toolbox.register('mutInsert', gp.mutInsert, pset=pset)
    
    mutOperators = [ toolbox.mutUniform, toolbox.mutEphemeral, toolbox.mutNodeReplacement, toolbox.mutInsert ]
    
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
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean) #using nanmean will hide nans that ARE in the pop, not ideal
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    eaElite(pop, toolbox, cxpb=1, mutpb=0.25, ngen=500, nelite=1, stats=stats, halloffame=hof)

    # print("\nBest Hof:\n%s"%hof[0])

    return pop, stats, hof


if __name__ == "__main__":
    main()
    # print("\nTime To Evo: %0.2f"%timeit.timeit(stmt=main, number=1))
