#!/usr/bin/env python3

# Does the neccessary imports
import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Time table maker.")
# Allows user input for which results file to use and which student to use
parser.add_argument(
    "-r",
    "--results",
    default="results/custom,small_data.npz,custom,0.01,50,2,100,False,1,1.npz",
    help="Which results file",
)

parser.add_argument(
    "-t",
    "--person_time_table",
    type=int,
    help="Print time table for student #",
)

args = parser.parse_args()

# Obtains the results from the results file
print("Loading", args.results)
results = np.load(args.results)
print("Loading", str(results["input"]))

# Then obtains all the data from the results file
inputs = np.load(str(results["input"]))
matrix_lecture_people = inputs["lecture_person"]
solution = results["best_solution"]

num_days = int(inputs["num_days"])
num_hours_per_day = int(inputs["num_hours_per_day"])

# Creates a function to generate a timetable
def person_time_table(person):
    print("Timetable for person", person)
    print("-----------------------------------")

    # Creates an empty matrix to store that persons lectures
    persons_lectures = []

    # Then finds each lecture that person should be attending, and places it in the previously created matrix
    for i in range(0, matrix_lecture_people.shape[0]):
        if matrix_lecture_people[i, person] != 0:
            persons_lectures.append(i)

    # Creates an empty list to store the timeslots
    timetable = []

    # Loops through each lecture the person has
    for lecture in persons_lectures:

        # Then loops through each room, and looks for a non zero timeslot
        for room in range(0, solution.shape[0]):
            timeslot = solution[room, lecture]

            if timeslot != 0:
                # Once a non zero timeslot is found, the timeslot,room and lecture are stored in the timetable list
                timetable.append((timeslot - 1, lecture, room))

    # This list is then sorted by timeslot
    timetable = sorted(timetable)

    # Then each entry in the timetable list is looped through, and printed to show the persons room and lecture, and when they take place
    for timeslot, lecture, room in timetable:
        day = timeslot // num_hours_per_day
        hour = timeslot % num_hours_per_day
        print(f"Day: {day+1} Hour: {hour+1} Lecture: {lecture+1:2d} Room: {room:2d}")

    print("-----------------------------------")


person_time_table(args.person_time_table)
