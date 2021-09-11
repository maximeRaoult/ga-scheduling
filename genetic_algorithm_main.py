#!/usr/bin/env python3

# All used imports, argpass to run from terminal with arguments,
# random for the relevant random functions such as mutations,
# and time to time how long trials were
import argparse
import random
import time

# numpy for matrices and linear algebra
# pygad for the genetic algorithms, and costs contains the relevant cost functions
# tqdm to time trials and have a progress bar to see how long running would take ahead of time
import numpy as np
import pygad
from tqdm import tqdm

from costs import *

parser = argparse.ArgumentParser(description="Genetic Algorithm Scheduler")

# Each of the posible arguments and hyperparameters that can be passed into the algorithm, can be entered as arguments
parser.add_argument(
    "-c",
    "--crossover",
    default="custom",
    help="The crossover function",
)

parser.add_argument(
    "-r",
    "--random_population",
    action="store_true",
    help="Whether the population wont be weighted initially",
)

parser.add_argument(
    "-m",
    "--mutation",
    default="custom",
    help="The crossover function",
)

parser.add_argument(
    "-p",
    "--num_parents_mating",
    type=int,
    default=2,
    help="Number of parent mating",
)

parser.add_argument(
    "-s",
    "--population_size",
    type=int,
    default=100,
    help="Number of solutions per population",
)

parser.add_argument(
    "-g",
    "--num_generations",
    type=int,
    default=10,
    help="Number of generations",
)

parser.add_argument(
    "-P",
    "--mutation_chance",
    type=float,
    default=0.01,
    help="Chance of a mutations",
)

parser.add_argument(
    "-i",
    "--input",
    default="small_data.npz",
    help="Inputs",
)

parser.add_argument(
    "-v",
    "--cost",
    default="1",
    help="WHich cost function",
)

args = parser.parse_args()

# Defining the fitness function as the negative of the cost function
def fitness(solution, solution_index):
    # Cost functions expect the solutions passed to be a matrix, rather then the 1D vectore pygad uses,
    # so this reshape the solutions back into the matrices
    solution = solution.reshape((num_rooms, num_lectures))

    # The cost functions are stored in costs.py, hence the earlier import
    return -cost(solution, LP)


# Generates a random gene, weighted towards zero, based on what value prob_zero has
# For example, if prob_zero=0.5, this would assign the gene a value of zero 50% of the time, and otherwise any possible timeslot
def random_genes(G, i, j):
    if random.random() < prob_zero:
        G[i, j] = 0
    else:
        G[i, j] = random.randint(1, num_timeslots)


# The custom weighted mutation function described in the paper
def mutation_func(offspring, ga_instance):
    # Obtains how many genes are to be mutated
    num_mutations = ga_instance.mutation_num_genes

    # Loops through each of the offsprings to be mutated
    for chromosome_idx in range(offspring.shape[0]):

        # Performs this loop a number of times equal to how many mutations need to occur
        for i in range(num_mutations):
            # Picks a random gene in the solution
            random_gene_idx = random.randrange(0, offspring.shape[1])

            # Randomises the gene using the random_gene function shown prior
            random_genes(offspring, chromosome_idx, random_gene_idx)
    return offspring


# The custom crossover function described in the paper
def crossover_func(parents, offspring_size, ga_instance):

    # Creates an empty matrix of the size neccessary to store the number of offspring being made
    offspring = np.empty(shape=offspring_size)

    # Loops through each row of offspring, filling them out with offspring with each loop
    for i in range(offspring_size[0]):

        # Takes two random parents. If the same parent is chosen twice the offspring will just be a clone
        p1 = random.randrange(0, parents.shape[0])
        p2 = random.randrange(0, parents.shape[0])

        # Turn the chosen parents back into matrics from 1D vectors
        parent1 = parents[p1, :].reshape(num_rooms, num_lectures)
        parent2 = parents[p2, :].reshape(num_rooms, num_lectures)

        # Create an empty matrix for the child
        child = np.empty(shape=(num_rooms, num_lectures))

        # Loops through each column of the parents, selecting each from a randomly selected parent
        for j in range(parent1.shape[1]):

            if random.random() < 0.5:
                child[:, j] = parent1[:, j]
            else:
                child[:, j] = parent2[:, j]

        # Returning the created offspring back into the 1D vector that pygad expects
        offspring[i, :] = child.flatten()

    return offspring


# Creating 2-point crossover. Single-point crossover is built into pygad so no need to implement it manually
def two_point(parents, offspring_size, ga_instance):

    # Creates an empty matrix of the size neccessary to store the number of offspring being made
    offspring = np.empty(shape=offspring_size)

    for i in range(offspring_size[0]):

        # Takes two random parents. If the same parent is chosen twice the offspring will just be a clone
        p1 = random.randrange(0, parents.shape[0])
        p2 = random.randrange(0, parents.shape[0])

        parent1 = parents[p1, :]
        parent2 = parents[p2, :]

        # Create an empty matrix for the child
        child = np.empty(shape=(parent1.shape))

        # Select two random split points
        split1 = random.randrange(1, parent1.shape[0] - 1)
        split2 = random.randrange(split1 + 1, parent1.shape[0])

        # Combine the parents together based on the calculated split points
        child = np.concatenate(
            (parent1[:split1], parent2[split1:split2], parent1[split2:]), axis=0
        )

        # Placing the generated offspring into the offspring matrix
        offspring[i, :] = child.flatten()

    return offspring


# In order to update the progressbar, and to keep track of the average fitnessess of each generation.
# This will be run every time a generation has been calculated
def collect_stats(ga):
    pbar.update(1)
    avg.append(np.mean(ga.last_generation_fitness))
    std.append(np.std(ga.last_generation_fitness))


# Load the dataset
inputs = np.load(args.input)

# Obtain the different parameters from the input. LP is the Lecture-Person table
LP = inputs["lecture_person"]

num_lectures = int(inputs["num_lectures"])
num_rooms = int(inputs["num_rooms"])
num_timeslots = int(inputs["num_timeslots"])

# Calculate some missing parameters. The prob_zero here is the same calculation mentioned in the paper in chapter 4.4
num_genes = num_lectures * num_rooms
prob_zero = 1 - num_lectures / (num_rooms * num_lectures)

# Sets the correct cost function based on the input from the user
if args.cost == "1":
    cost = cost_v1
else:
    cost = cost_v2

# Changes the parameters to be passed into the genetic algorithm if a initial population with genes weighted towards zero is desired
if args.random_population:
    # This is run if no weighting is desired
    sol_per_pop = args.population_size
    initial_population = None
else:
    # Otherwise, the initial population is created as a set of zero matrices, then filled in using the random_genes function from earlier
    sol_per_pop = None
    initial_population = np.zeros((args.population_size, num_genes))
    for i in range(0, args.population_size):
        for j in range(0, num_genes):
            random_genes(initial_population, i, j)

# Sets the crossover function based on the users inputs
if args.crossover == "custom":
    crossover = crossover_func
elif args.crossover == "two_point":
    crossover = two_point
else:
    crossover = args.crossover

# Sets the mutation function based on the users inputs
mutation = mutation_func if args.mutation == "custom" else args.mutation

# Starts the set of averages to be empty for the collect_stats function
avg = []
std = []

# Sets the start time
start = time.time()

# Creates the progressbar
with tqdm(total=args.num_generations) as pbar:
    # Runs the genetic algorith with each of the parameters chosen
    ga_instance = pygad.GA(
        num_generations=args.num_generations,
        num_parents_mating=args.num_parents_mating,
        sol_per_pop=sol_per_pop,
        initial_population=initial_population,
        mutation_num_genes=round(num_genes * args.mutation_chance),
        mutation_type=mutation,
        crossover_type=crossover,
        num_genes=num_genes,
        fitness_func=fitness,
        gene_type=int,
        gene_space=dict(low=0, high=num_timeslots + 1),
        on_generation=collect_stats,
    )

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

# Calculates how much time elapsed during running
elapsed = time.time() - start

# Obtains the best solution found, and reshapes it back into a matrix
solution, solution_fitness, solution_index = ga_instance.best_solution()
solution = solution.reshape((num_rooms, num_lectures))

# Calculates the costs on this solution, however also outputs how often each constraint was broken by this solution
cost(solution, LP, True)

# Generates a filename based on all of the entered parameters
results_file = f"results/{args.crossover},{args.input},{args.mutation},{args.mutation_chance},{args.num_generations},{args.num_parents_mating},{args.population_size},{args.random_population},{args.cost}.npz"
# Saves all the data on the genetic algorithm and its results into this file
np.savez(
    results_file,
    time_taken=elapsed,
    averages=avg,
    standard_dev=std,
    best_solutions=ga_instance.best_solutions_fitness,
    crossover=args.crossover,
    input=args.input,
    mutation=args.mutation,
    mutation_chance=args.mutation_chance,
    num_generations=args.num_generations,
    num_parents_mating=args.num_parents_mating,
    population_size=args.population_size,
    random_population=args.random_population,
    cost=args.cost,
    best_solution=solution,
)
