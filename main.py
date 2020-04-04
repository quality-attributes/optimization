import argparse
import logging
import numpy
import random
import sys


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

__author__ = "Manolomon"
__license__ = "MIT"
__version__ = "1.0"

import pickle

def fitness_func(individuo): # Fitness Function
    global X_train
    global y_train
    classifier = MultinomialNB(alpha=individuo[0], 
                               fit_prior = False if individuo[1] < 0.5 else True)
    classifier.fit(X_train, y_train)

    acc = cross_val_score(classifier, X_train, y_train, cv=5)
    del classifier
    return acc.mean()

def reflection(vector): # Boundary constraint-handling
    global limits

    for i in range(len(limits)):
        param_range = limits[i]
        if vector[i] <= param_range[0]: 
            vector[i] = (2 * param_range[0]) - vector[i]
        if vector[i] >= param_range[1]:
            vector[i] = (2 * param_range[1]) - vector[i]
    return vector

def ed_rand_1_bin(np, max_gen, f, cr):
    global limits

    #print(str(np) + ' - ' + str(max_gen) + ' - ' + str(f) + ' - ' + str(cr))
    # Initialize population
    alphas = numpy.random.uniform(low=limits[0][0], high=limits[0][1], size=(np,1))
    fit_priors = numpy.random.uniform(low=limits[1][0], high=limits[1][1], size=(np,1))
    population = numpy.concatenate((alphas, fit_priors), axis=1)
    # First evaluation of population
    logging.debug("Start of evolution")
    fitness = numpy.apply_along_axis(fitness_func, 1, population)
    order = numpy.argsort(fitness)
    population = population[order]

    # Evolutionary process
    for g in range(max_gen):
        logging.debug("-- Generation %s --" % str(g + 1))
        for i in range (np):
            # Mutation
            no_parent = numpy.delete(population, i, 0)
            # Random pick of individuals
            row_i = numpy.random.choice(no_parent.shape[0], 3, replace=False)
            r = no_parent[row_i, :]
            v_mutation = ((r[0]-r[1]) * f) + r[2]
            # Reflection for boundaries constrain-handling
            v_mutation = reflection(v_mutation)
            # Crossover
            jrand = random.randint(0, 1)
            v_son = numpy.empty([1, 2])
            for j in range(2):
                if random.uniform(0, 1) < cr or j == jrand:
                    v_son[0,j] = v_mutation[j]
                else:
                    v_son[0,j] = population[i,j]
            population = numpy.concatenate((population, v_son), axis=0)
            # Reevaluation
            fitness = numpy.apply_along_axis(fitness_func, 1, population)
            order = numpy.argsort(fitness)[::-1]
            population = population[order]
        logging.debug("Current best internal CV score: [alpha=%s, fit_prior=%s], fitness=%s" % (alphas[0], False if fit_priors[0] < 0.5 else True, fitness[0]))
        # Surplus disposal
        population = population[:np]
        fitness = fitness[:np]
    return fitness[0]

def main(np, max_gen, f, cr):
    score = ed_rand_1_bin(np, max_gen, float(f), float(cr))

if __name__ == "__main__":
        
    ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--np', dest='np', type=int, required=True, help='Population size')
    ap.add_argument('--max_gen', dest='max_gen', type=int, required=True, help='Generations')
    ap.add_argument('--f', dest='f', type=float, required=True, help='Scale Factor')
    ap.add_argument('--cr', dest='cr', type=float, required=True, help='Crossover percentage')
    ap.add_argument('-e', "--execution", type=int,required=True, help='Execution number')

    with open('../text-representation/X_train.pickle', 'rb') as data:
        X_train = pickle.load(data)
    with open('../text-representation/y_train.pickle', 'rb') as data:
        y_train = pickle.load(data)

    limits = [[0,100],[0,1]]

    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(filename=('logfile_' + str(args.execution) + '.log'),
                            filemode='a',level=logging.DEBUG)

    logging.debug(args)
    main(args.np, args.max_gen, args.f, args.cr)

