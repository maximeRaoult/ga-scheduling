#!/usr/bin/env python3

# Does all needed imports
import argparse
import os
import random

import numpy as np
import yaml

parser = argparse.ArgumentParser(description="Data creator")

# Lets you add what dataset you wish to create based on which settings file you pass
parser.add_argument(
    "-i",
    "--input",
    default="small_data.yaml",
    help="Inputs",
)

args = parser.parse_args()

# Obtains the file that is desired to be used for the dataset
path = args.input
with open(path, "r") as f:
    data = yaml.safe_load(f)

# Obtaines all the different parameters in the file
num_students = data["num_students"]
num_rooms = data["num_rooms"]
num_profs = data["num_profs"]

num_days = data["num_days"]
num_hours_per_day = data["num_hours_per_day"]
num_modules = data["num_modules"]

num_lectures_per_module = data["num_lectures_per_module"]
num_module_per_student = data["num_module_per_student"]

# Calculates some needed values that are not stored in the file
num_lectures = num_modules * num_lectures_per_module
num_timeslots = num_days * num_hours_per_day

# Creates thge list of modules
i = 0
modules = []
for module in range(num_modules):
    # For example if there are three lectures per module, this will create a module with lectures 0,1,2, then
    # one with lectures 3,4,5, and so on until all modules have thier corresponding number of lectures
    modules.append(list(range(i, i + num_lectures_per_module)))
    i += num_lectures_per_module

# Generates an empty matrix that stores each students modules and lectures
lecture_students = np.zeros((num_lectures, num_students))
module_students = np.zeros((num_modules, num_students))
all_modules = list(range(num_modules))

# Loops through each student
for student in range(num_students):
    # Assigns each student a random set of modules, based on how many modules each student should have
    rand_mods = random.sample(all_modules, num_module_per_student)

    # Then fills in the two matrices, lecture_students and module_students, with values of one for each module and lecture they have
    for mod in rand_mods:
        module_students[mod, student] = 1
        for lect in modules[mod]:
            lecture_students[lect, student] = 1

# Generates an empty matrix that stores each lecturer's (called profs here to be less confusing) lectures
lecture_profs = np.zeros((num_lectures, num_profs))

# Then loops through each lecture
for lect in range(num_lectures):
    # And assigns that lecture to a random lecturer, and fills in the matrix acordingly
    rand_prof = random.randrange(0, num_profs)
    lecture_profs[lect, rand_prof] = 1

# Concatinates the student and lecturer matrices together, to create the single Lecture-Person matrix
lecture_person = np.hstack((lecture_students, lecture_profs))

# Obtains the user entered filename
fname, ext = os.path.splitext(path)
# Creates the file storing all this data using the same name as the .yaml file which contained the file, and saves it all as an .npz
np.savez(
    fname + ".npz",
    lecture_person=lecture_person,
    module_students=module_students,
    lecture_profs=lecture_profs,
    lecture_students=lecture_students,
    num_lectures=num_lectures,
    num_timeslots=num_timeslots,
    num_rooms=num_rooms,
    num_students=num_students,
    num_profs=num_profs,
    num_days=num_days,
    num_hours_per_day=num_hours_per_day,
)

print("Saved data to " + fname + ".npz")
