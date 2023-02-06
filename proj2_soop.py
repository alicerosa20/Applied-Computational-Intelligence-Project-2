import random

import numpy as np
from deap import base
from deap import creator
from deap import tools
import pandas as pd
from random import sample
import matplotlib.pyplot as plt


# Start by uncoment the number of cities and objective function you want to consider for the problem

Points_Cities = 20
#Points_Cities = 30
#Points_Cities = 50

cost  = (pd.read_csv('CityDistCar.csv', sep=',', decimal='.', usecols=[*range(1,Points_Cities+1)])).to_numpy()
#cost = (pd.read_csv('CityDistPlane.csv', sep=',', decimal='.', usecols=[*range(1,Points_Cities+1)])).to_numpy()
#cost = (pd.read_csv('CityCostCar.csv', sep=',', decimal='.', usecols=[*range(1,Points_Cities+1)])).to_numpy()
#cost = (pd.read_csv('CityCostPlane.csv', sep=',', decimal='.', usecols=[*range(1,Points_Cities+1)])).to_numpy()
coord = (pd.read_csv('CitiesXY.csv', sep=',', decimal='.')).to_numpy()


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

N_CITIES = range(0,Points_Cities)

def shuffle_vetor():
    list_cities = list(N_CITIES)

    random.shuffle(list_cities)
    return creator.Individual(list_cities)

toolbox.register("individual", shuffle_vetor)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be minimized
def path_ev(individual, cost):

    total_cost=0

    for i in range(len(individual)-1):
        total_cost = total_cost + cost[individual[i],individual[i+1]]
    
    total_cost = total_cost + cost[individual[len(individual)-1],individual[0]]

    return total_cost,

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", path_ev, cost=cost)

# register the crossover operator
toolbox.register("mate", tools.cxPartialyMatched)

def inversion(individual):
    subset = sample(N_CITIES, 2)
    subset.sort()
    int = subset[1]-subset[0]
    count = 0

    for i in range(subset[0], subset[0]+round(int/2)+1):
        a=individual[i]
        b=individual[subset[1]-count]
        individual[i]=b
        individual[subset[1]-count]=a
        count = count + 1
    
    return individual


# register a mutation operator 
toolbox.register("mutate", inversion)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=50)
 
def heuristica():

    x_city_left = min(coord[:Points_Cities,1])
    x_city_right = max(coord[:Points_Cities,1])
    middle = (x_city_left+x_city_right)/2

    cities_left = coord[:Points_Cities][coord[:Points_Cities,1]<middle]
    cities_right = coord[:Points_Cities][coord[:Points_Cities,1]>middle]

    cities_left = cities_left[cities_left[:, 2].argsort()]
    cities_right = cities_right[cities_right[:, 2].argsort()[::-1]]

    path = np.concatenate([cities_left[:,0], cities_right[:,0]])

    return creator.Individual(path.tolist())



def EA_App(n_seed):

    random.seed(n_seed)

    # create an initial population of 60 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=60)

    # Heuristica 
    # path = heuristica()
    # ind = random.randrange(len(pop))
    # pop[ind]=path

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.6, 0.15

    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    best_fitness=[]

    # Begin the evolution
    while g < 200:
        # A new generation
        g = g + 1

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        best_ind = tools.selBest(pop, 1)[0]
        best_fitness.append(best_ind.fitness.values)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    return best_ind, best_ind.fitness.values, best_fitness


def main():

    best_fit=[]
    curr_best_fit = 10**(5)

    for i in range(30):

        print("-- Run %i --" % (i+1))
        a,b,c = EA_App(35+i)

        best_fit.append(b)

        if b[0] < curr_best_fit:
            curr_best_fit = b
            curr_best_path = a
            curr_gen_fit = c
    
    best_fit = np.array(best_fit)
    mean = sum(best_fit) / len(best_fit)
    var = sum((x-mean)**2 for x in best_fit) / len(best_fit)
    std = var**0.5
    print("\n -- Final Review --")
    print(" Mean of 30 runs: %s" % mean)
    print(" Std  of 30 runs: %s" % std)
    print(" Best individual is %s, %s" % (curr_best_path, curr_best_fit))

    best_fitness = np.array(curr_gen_fit)
    plt.plot([*range(1,best_fitness.size+1)], best_fitness)
    plt.xlabel('#Generations')
    plt.ylabel('Cost')
    plt.show()


if __name__ == "__main__":
    main()



