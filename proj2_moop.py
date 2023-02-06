import array
import random
import json

import numpy
from matplotlib import pyplot as plt

from math import sqrt
from numpy.core.fromnumeric import argmax

import pandas as pd
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from random import sample

# Start by uncoment the number of cities you want to consider for the problem

Points_Cities = 20
#Points_Cities = 30
#Points_Cities = 50

coord = (pd.read_csv('CitiesXY.csv', sep=',', decimal='.')).to_numpy()
cost_car = (pd.read_csv('CityCostCar.csv', sep=',', decimal='.', usecols=[*range(1,Points_Cities+1)])).to_numpy()
cost_plane = (pd.read_csv('CityCostPlane.csv', sep=',', decimal='.', usecols=[*range(1,Points_Cities+1)])).to_numpy()
dist_car  = (pd.read_csv('CityDistCar.csv', sep=',', decimal='.', usecols=[*range(1,Points_Cities+1)])).to_numpy()
dist_plane = (pd.read_csv('CityDistPlane.csv', sep=',', decimal='.', usecols=[*range(1,Points_Cities+1)])).to_numpy()

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

N_CITIES = range(0,Points_Cities)

def shuffle_vetor():
    list_cities = list(N_CITIES)
    count = 0
    random.shuffle(list_cities)
    transport = []
    
    for i in range(0,Points_Cities):
        n = random.randint(0,1) # Car -> 0; Plane -> 1

        # Apply hard constraint
        if n==1: 
            count = count+1
        elif n==0 and count!=0:
            count = 0
        
        if count==4:
            n=0

        transport.append(n)

    return creator.Individual(list(map(list, zip(list_cities,transport))))

toolbox.register("individual", shuffle_vetor)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def path_ev(individual):

    total_cost=0
    total_time=0

    for i in range(len(individual)-1):
    
        if individual[i][1]==0:
            total_cost = total_cost + cost_car[individual[i][0],individual[i+1][0]]
            total_time = total_time + dist_car[individual[i][0],individual[i+1][0]]
        elif individual[i][1]==1:
            total_cost = total_cost + cost_plane[individual[i][0],individual[i+1][0]]
            total_time = total_time + dist_plane[individual[i][0],individual[i+1][0]]
    
    if individual[len(individual)-1][1]==0:
        total_cost = total_cost + cost_car[individual[len(individual)-1][0],individual[0][0]]
        total_time = total_time + dist_car[individual[len(individual)-1][0],individual[0][0]]
    elif individual[len(individual)-1][1]==1:
        total_cost = total_cost + cost_plane[individual[len(individual)-1][0],individual[0][0]]
        total_time = total_time + dist_plane[individual[len(individual)-1][0],individual[0][0]]

    return total_cost,total_time
    
def apply_crossover(ind1,ind2):
    
    cities_1 = [item[0] for item in ind1]
    transport_1 = [item[1] for item in ind1]
    cities_2 = [item[0] for item in ind2]
    transport_2 = [item[1] for item in ind2]
    
    tools.cxPartialyMatched(cities_1,cities_2)
    tools.cxPartialyMatched(transport_1,transport_2)

    for i in range(len(ind1)):
        ind1[i]=[cities_1[i], transport_1[i]]
    
    for i in range(len(ind1)):
        ind2[i]=[cities_2[i], transport_2[i]]

    return ind1,ind2

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

def apply_mutation(ind):
    cities = [item[0] for item in ind]
    transport = [item[1] for item in ind]

    inversion(cities)
    inversion(transport)

    for i in range(len(ind)):
        ind[i]=[cities[i], transport[i]]

    return ind

def validate_hard_const(ind1, ind2):
    count = 0

    for i in range(len(ind1)):
        n=ind1[i][1]
        
        if n==1: 
            count = count+1
        elif n==0 and count!=0:
            count = 0
        
        if count==4:
            ind1[i][1]=0
    
    count = 0
    for j in range(len(ind2)):
        n=ind2[j][1]
        
        if n==1: 
            count = count+1
        elif n==0 and count!=0:
            count = 0
        
        if count==4:
            ind2[j][1]=0
    
    return ind1, ind2


toolbox.register("evaluate", path_ev)
toolbox.register("mate", apply_crossover)
toolbox.register("mutate", apply_mutation)
toolbox.register("select", tools.selNSGA2)

def main(seed=64):
    random.seed(seed)

    NGEN = 250
    MU = 100
    CXPB = 0.9
    CMUT = 0.2

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pareto_front = tools.ParetoFront()
    
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    hypervolume_values = []
    pop_values = numpy.array([ind.fitness.values for ind in pop])
    ref = [max(pop_values[:,0]),max(pop_values[:,1])]

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            
            if random.random() <= CMUT:
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

            validate_hard_const(ind1,ind2)

            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        pareto_front.update(pop)
        pareto_values = numpy.array([ind.fitness.values for ind in pareto_front.items])
        hypervolume_values.append(hypervolume(pareto_front, ref))
        print(logbook.stream)
        

    plt.plot(pareto_values[:,0],pareto_values[:,1])
    plt.scatter(pareto_values[:,0],pareto_values[:,1], c='r')
    plt.xlabel('Cost')
    plt.ylabel('Dist/Time')
    plt.title('Pareto Curve')
    plt.show()
    plt.plot(list(range(1, NGEN)), hypervolume_values)
    plt.xlabel('#Generations')
    plt.ylabel('Hypervolume')
    plt.show()


        
if __name__ == "__main__":
    main()