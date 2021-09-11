#!/usr/bin/env python3
# Does the neccessary imports
import os
import subprocess

# This file allows you to run the genetic algorithm with easier access to the settings,
# rather then typing them all out in ther terminal


# Here each of the settings can be changed as desired
crossover = "custom"
# crossover = "single_point"
# crossover = "two_point"

mutation = "custom"
# mutation = "random"

input = "small_data.npz"
# input="medium_data.npz"

mutation_chance = 0.01

num_generations = 50

num_parents_mating = 2

population_size = 100

# False for weighted genes in begining population, True for complete randomness in the genes of begining population
random_population = False
# random_population = True

# 1 for not normalised cost funtion
# 2 for normalised
cost = 1

# Generates the filename based on the settings
results = f"results/{crossover},{input},{mutation},{mutation_chance},{num_generations},{num_parents_mating},{population_size},{random_population},{cost}.npz"

# Checks if the file already exists or not
if not os.path.exists(results):
    # Creates a string which stores a terminal command that will run the genetic algorithm using the settings chosen above
    cmd = [
        "python3",
        "./genetic_algorithm_main.py",
        "--crossover",
        crossover,
        "--input",
        input,
        "--mutation",
        mutation,
        "--mutation_chance",
        str(mutation_chance),
        "--num_generations",
        str(num_generations),
        "--num_parents_mating",
        str(num_parents_mating),
        "--population_size",
        str(population_size),
        "--cost",
        str(cost),
    ]

    if random_population:
        cmd.append("--random_population")

    # Runs the command in the terminal
    subprocess.call(cmd)
