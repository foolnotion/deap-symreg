import random
import numpy as np
from inspect import isclass

from deap import algorithms
from deap import tools
from deap import gp


def genBalanced(pset_, depth_, dist_, dist_args_, type_=None):
    length_ = int(np.round(dist_(*dist_args_)))
    return btc(pset_, depth_, length_, type_)


def btc(pset_, depth_, length_, type_=None):
    if type_ is None:
        type_ = pset_.ret

    expr = []

    arities = list(map(lambda x: x.arity, pset_.primitives[type_]))
    minFunctionArity = min(arities)
    maxFunctionArity = max(arities)

    # adapt length to restrictions of the primitive set
    if length_ % 2 == 0 and minFunctionArity > 1:
        length_ = length_ + 1 if np.random.random_sample(1) > 0.5 else length_ - 1

    targetLength = length_ - 1 # don't count the root node 
    maxFunctionArity = min(maxFunctionArity, targetLength)
    minFunctionArity = min(minFunctionArity, targetLength)
    root = sampleChild(pset_, minFunctionArity, maxFunctionArity, type_) 

    # inner lists of the form [node, depth, childIndex] 
    # childIndex is only used at the end to transform 
    # the representation from breadth to prefix
    expr.append([root, 0, 1])

    openSlots = root.arity 

    for i in range(0, length_):
        (node, nodeDepth, childIndex) = expr[i]
        childDepth = nodeDepth + 1
        
        for j in range(0, getArity(node)):
            maxArity = 0 if childDepth == depth_ - 1 else min(maxFunctionArity, targetLength - openSlots)
            minArity = min(minFunctionArity, maxArity)
            child = sampleChild(pset_, minArity, maxArity, type_)

            if j == 0:
                expr[i][2] = len(expr)

            expr.append([child, childDepth, 0])
            openSlots += getArity(child) 

    nodes = breadthToPrefix(expr)
    return nodes


def sampleChild(pset_, minArity_, maxArity_, type_=None):
    candidates = []

    if maxArity_ > 0:
        for prim in pset_.primitives[type_]:
            if prim.arity > maxArity_:
                continue
            candidates.append(prim)

    if minArity_ == 0:
        # consider terminals awe sll
        terminals = pset_.terminals[type_]
        p = 1 / (len(candidates) + 1)
        if np.random.binomial(1, p, 1):
            term = random.choice(terminals)
            if isclass(term):
                term = term()
            return term

    return random.choice(candidates)


def breadthToPrefix(expr):
    prefix = []

    def addPrefix(t):
        node, _, index = t
        prefix.append(node)
        for i in range(index, index + getArity(node)):
            addPrefix(expr[i])

    addPrefix(expr[0])
    return prefix 


def getArity(node):
    return node.arity if isinstance(node.arity, int) else 0

