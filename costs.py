# Does the neccessary imports
from collections import defaultdict

import numpy as np


# Defines each penalty function similar to how it was described in the paper, this one checks each lecture has been scheduled exactly once
def lecture_penalties(X, report=False):
    # Sets the initial penalties to be zero, and keeps track of broken constraints
    penalty = 0
    num_of_dupes = 0
    not_scheduled = 0

    # This loops through each lecture
    for j in range(X.shape[1]):

        # Counts how many non-zero values each lecture has in its corresponding column
        count = np.count_nonzero(X[:, j])

        # If a report of broken constraints is desired, this will keep count
        if report:
            if count != 0:
                num_of_dupes += count - 1
            else:
                not_scheduled += 1
        # Calculates the penalty, punishing any value that is not exactly one, since we only want one non zero value in each column
        penalty += (1 - count) ** 2

    # Prints the report if it was desired
    if report:
        print(f"Number of double booked lectures: {num_of_dupes}")
        print(f"Number of non scheduled lectures: {not_scheduled}")

    return penalty


# This checks if a room has been booked multiple times at the same timeslot
def doublebook_penalties(X, report=False):
    # Sets the initial penalties to be zero
    penalty = 0

    # Loops through each room
    for i in range(X.shape[0]):

        # Creates a list to keep track of which timeslots the room has been assigned
        S = []

        # Loops through each lecture
        for j in range(X.shape[1]):
            # Obtains the timeslot that the lecture j will be in room i
            x = X[i, j]

            # Check if the timeslot is zero, since if it is, this means that there is no booking at all
            if x != 0:

                # Applies a penalty if the timeslot has already been recorded in S, otherwise adds it to S
                if x in S:
                    penalty += 1
                else:
                    S.append(x)

    # Prints the report if it was desired
    if report:
        print(f"Number of double booked rooms: {penalty}")

    return penalty ** 2


# This penalty checks for a person being in two lectures at once, or not having one of thier lectures booked
def people_double_booking(X, matrix_lecture_people, report=False):
    # Keeps track of broken constraints
    missing_lectures = 0
    duplicate_times = 0

    # First it loops through each person
    for j in range(0, matrix_lecture_people.shape[1]):

        # Creates an empty matrix to store that persons lectures
        persons_lectures = []

        # Then finds each lecture that person should be attending, and places it in the previously created matrix
        for i in range(0, matrix_lecture_people.shape[0]):
            if matrix_lecture_people[i, j] != 0:
                persons_lectures.append(i)

        # Creates a dictionary which will store how often each timeslot is found, starting at zero
        timeslots_of_lectures = defaultdict(int)

        # Starts looping through the current persons lectures
        for l in persons_lectures:

            # Will keep track of the first non-zero timeslot found. If a lecture has been scheduled more then once, we simply take the first timeslot in its column
            first_timeslot = 0

            # This finds the first non-zero timeslot
            for i in range(0, X.shape[0]):
                if X[i, l] != 0:
                    first_timeslot = X[i, l]
                    break

            # If first timeslot is still zero, it means the lectures has not been booked, so the penalty is applied
            if first_timeslot == 0:
                missing_lectures += 1
            else:
                # Otherwise, the corresponding entry in the dictionary is incremented by one
                timeslots_of_lectures[first_timeslot] += 1

        # Then it sums over all non-zero values in the dictionary minus one, since the first time a
        # person is scheduled a lecture at a time does not break the constraint, only for each repeated scheduling
        for c in timeslots_of_lectures.values():
            duplicate_times += c - 1

    # Prints the report if it was desired
    if report:
        print(f"Number of missing lectures (for people): {missing_lectures}")
        print(f"Number of duplicate lectures (for people): {duplicate_times}")

    return (missing_lectures + duplicate_times) ** 2


# This sums the costs each of the penalty functions above returned
def cost_v1(X, LP, report=False):
    return (
        doublebook_penalties(X, report)
        + lecture_penalties(X, report)
        + people_double_booking(X, LP, report)
    )


# These are the highest values found for each penalty after testing
wr = 1 / 81
wl = 1 / 149
wp = 1 / 85803169

# This is the same cost function as before, however the penalties are normalised between zero and one using the values found
def cost_v2(X, LP, report=False):

    a = doublebook_penalties(X, report)
    b = lecture_penalties(X, report)
    c = people_double_booking(X, LP, report)

    return a * wr + b * wl + c * wp
