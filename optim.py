"""Compute additional data required by and run the optimization algorythm.

All functions defined here use the data processed by `data_preprocessing`.
Some compute additional information from them, and some define the Mixed
Integer model, and solves it using SCIP and it's PySCIPOpt python interface.
"""

import pandas as pd
import numpy as np
from pyscipopt import Model
from pyscipopt import quicksum


VALUE_FIRST = 1
VALUE_SECOND = -1


def gen_utility(prefs):
    """Compute the utility values that will go into the objective function.

    prefs: (pd.DataFrame) first and second choices of people for every time
    slot.
    """
    first = prefs["first"]
    second = prefs["second"]
    # empty dataframe for storing the utilities
    utility = pd.DataFrame(index=prefs.index, columns=first.columns)
    # set VALUE_SECOND on second choices
    utility.where(~(second == 1), VALUE_SECOND, inplace=True)
    # set VALUE_FIRST on first choices (override second choices if both are provided)
    utility.where(~(first == 1), VALUE_FIRST, inplace=True)

    return utility


def nb_train_asked(utility):
    """Compute the number of trainings people asked for (in the week).

    The number of trainings people asked for is their number of first choice.
    """
    return utility.apply(lambda x: x.value_counts()[VALUE_FIRST], axis=1)


def boats_for_people(people, boats, match):
    """Compute the boats that each person can row.

    people: list of people with their skills and weight category,
    boats: list of boats with their class and weight category
    Returns:

    boats_for_person: dict with each person as keys, and the set of boats they
    can row based on their skills and weight class.
    """
    boats_for_person = dict()
    for p in people.index:
        skill = people.loc[p]["skill"]
        weight = people.loc[p]["weight"]

        # mask selecting boats with a skill mathcing the level of the person
        skill_mask = boats["boat_class"].isin(match[skill])
        # mask selecting boats that match the weight class of the person
        weight_mask = boats[weight] == 1

        matching_boats = boats[skill_mask & weight_mask]
        boats_for_person[p] = set(matching_boats.index)

    return boats_for_person


def times_for_people(utility):
    times_for_person = dict()
    for p in utility.index:
        # available times for the person
        times_for_person[p] = set(utility.loc[p][utility.loc[p].notna()].index)

    return times_for_person


def people_for_boat_time(people, boats, utility, reverse_match):
    # people likely to row a given boat at a given time
    people_for_boat_time = dict()
    for b in boats.index:
        skill_mask = pd.Series(data=False, index=people.index)
        for skill in reverse_match[boats.loc[b]["boat_class"]]:
            skill_mask |= people["skill"] == skill

        weight_mask = pd.Series(
            data=False, index=people.index
        )  # neutral mask (does not mask anything)
        for weight in ["L", "M", "MH", "H"]:
            if boats.loc[b][weight] == 1:
                weight_mask |= people["weight"] == weight

        people_for_boat = set(people[skill_mask & weight_mask].index)

        for t in utility.columns:
            people_for_boat_time[(b, t)] = {
                p for p in people_for_boat if not np.isnan(utility.loc[p, t])
            }

    return people_for_boat_time


def define_model(
    utility,
    nb_train_asked,
    boats,
    boats_for_person,
    times_for_person,
    people_for_boat_time,
):
    # definition of the model
    model = Model("boat_alloc")

    people = utility.index
    # defining varialbes
    variables = {}
    for p in people:
        for b in boats_for_person[p]:
            for t in times_for_person[p]:
                variables[(p, b, t)] = model.addVar(vtype="B")

    # minimal pref score among all people, that we want to maximize
    s = model.addVar(vtype="I", name="min_pref", lb=None)

    # setting the objective fucntion
    total_pref = quicksum(
        utility.loc[p, t] * variables[(p, b, t)]
        for p in people
        for b in boats_for_person[p]
        for t in times_for_person[p]
    )
    model.setObjective(500 * s + total_pref, "maximize")

    # adding the constraints
    for p in people:
        # sum of prefs
        sum_of_prefs = quicksum(
            utility.loc[p, t] * variables[(p, b, t)]
            for b in boats_for_person[p]
            for t in times_for_person[p]
        )
        # s = min_{p} sum of prefs
        model.addCons(s <= sum_of_prefs)

        # each person trains the number of times they asked
        nb_trainings = quicksum(
            variables[(p, b, t)]
            for b in boats_for_person[p]
            for t in times_for_person[p]
        )
        model.addCons(nb_trainings <= nb_train_asked.loc[p])

        # each person can only row one boat at a time
        for t in times_for_person[p]:
            nb_boats = quicksum(variables[(p, b, t)] for b in boats_for_person[p])
            model.addCons(nb_boats <= 1)

        # not am1 and am2 in the same day
        # days where "am2" exists (i.e. not Saturday)
        weekdays = utility.columns[
            utility.columns.get_level_values("time") == "am2"
        ].unique("day")
        for day in weekdays:
            if not np.isnan(utility.loc[p, (day, "am1")]) and not np.isnan(
                utility.loc[p, (day, "am2")]
            ):
                model.addCons(
                    quicksum(
                        variables[(p, b, (day, "am1"))]
                        + variables[(p, b, (day, "am2"))]
                        for b in boats_for_person[p]
                    )
                    <= 1
                )

    # for each boat and time, no more than one person
    # for b in set(boats[boats["boat_class"].isin([0, 1, 2])].index):
    for b in boats.index:
        for t in utility.columns:
            nb_people_on_boat_time = quicksum(
                variables[(p, b, t)] for p in people_for_boat_time[(b, t)]
            )
            model.addCons(nb_people_on_boat_time <= 1)

    return model, variables, s


def optimize(
    model, variables, s, utility, nb_train_asked, boats_for_person, times_for_person
):

    model.optimize()

    status = model.getStatus()
    if status == "optimal":
        # generate resulting csv file with boat allocation
        result = pd.DataFrame(columns=utility.columns, index=utility.index)
        result.index.rename("people", inplace=True)

        fairness = pd.DataFrame(
            0, index=utility.index, columns=["nb_asked", "nb_first", "nb_second"]
        )
        fairness["nb_asked"] = nb_train_asked

        for p in utility.index:
            for t in times_for_person[p]:
                boat = {
                    b
                    for b in boats_for_person[p]
                    if model.isEQ(model.getVal(variables[(p, b, t)]), 1)
                }
                assert len(boat) <= 1
                if boat:
                    result.loc[p, t] = boat.pop()
                # if p is rowing at time t
                if model.isEQ(
                    model.getVal(
                        quicksum(variables[(p, b, t)] for b in boats_for_person[p])
                    ),
                    1,
                ):
                    if utility.loc[p, t] == VALUE_FIRST:
                        fairness.loc[p, "nb_first"] += 1
                    elif utility.loc[p, t] == VALUE_SECOND:
                        fairness.loc[p, "nb_second"] += 1
                    else:
                        raise

        fairness["diff"] = (
            fairness["nb_first"] + fairness["nb_second"] - fairness["nb_asked"]
        )

        # result.to_excel("results/comp/results_boat.xlsx")
        print("fairness score (0 = optimal) : ", model.getVal(s))

        return result, fairness

    else:
        print(f"solver finished with status {status}")

        return None, None
