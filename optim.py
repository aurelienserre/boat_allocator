import pandas as pd
import numpy as np
from pyscipopt import Model
from pyscipopt import quicksum


# import data
people = pd.read_csv("data_processed/people.csv", index_col="Name")
boats = pd.read_csv('data_processed/boats.csv', index_col='name')
time_prefs = {'comp': pd.read_csv('data_processed/comp/time_prefs.csv',
                                  header=[0, 1], index_col=0),
              'rec_master': None
              }


# Optim for competitive group
GROUP = 'comp'

prefs = time_prefs[GROUP]
people_set = set(prefs.index)
time_set = set(prefs.columns)
boat_set = set(boats.index)

# number of trainings people ask for (number of first choice)
asked_nb_trainings = prefs.apply(lambda x: x.value_counts()[0], axis=1)

# for each group, dict with key = level of people,
# values = list of boat skills they can row
boat_class_per_skill = {
    'comp': {
        1: [0, 1],
        2: [2]
    },
    'rec_master': {
        1: [1],
        2: [2],
        3: [2]
    }
}

# boats_for_person and times_avail_for_person for each person
boats_for_person = {}
times_avail_for_person = {}
for p in people_set:
    skill = people.loc[p]['Level']
    weight = people.loc[p]['Class']

    # mask selecting boats with a skill mathcing the level of the person
    skill_mask = boats['skill'].isin(boat_class_per_skill['comp'][skill])
    # mask selecting boats that match the weight class of the person
    weight_mask = boats[weight] == 1

    matching_boats = boats[skill_mask & weight_mask]
    boats_for_person[p] = set(matching_boats.index)

    # available times for the person
    times_avail_for_person[p] = set(prefs.loc[p][prefs.loc[p].notna()].index)

# people likely to row a given boat at a given time
people_for_boat_time = {}
for b in boat_set:
    skill_mask = people['membership'] == 'comp'
    if boats.loc[b]['skill'] in (0, 1):
        skill_mask &= people['Level'] == 1
    elif boats.loc[b]['skill'] == 2:
        skill_mask &= people['Level'] == 2
    else:
        continue

    weight_mask = pd.Series(data=False, index=people.index)  # neutral mask (does not mask anything)
    for weight in ['L', 'M', 'MH', 'H']:
        if boats.loc[b][weight] == 1:
            weight_mask |= people['Class'] == weight

    people_for_boat = set(people[skill_mask & weight_mask].index)
    # only people who gave time prefs
    people_for_boat &= people_set

    for t in time_set:
        people_for_boat_time[(b, t)] = {p for p in people_for_boat
                                        if not np.isnan(prefs.loc[p, t])}

# --------------
# definition of the model
model = Model("boat_alloc")

# defining varialbes
variables = {}
for p in people_set:
    for b in boats_for_person[p]:
        for t in times_avail_for_person[p]:
            variables[(p, b, t)] = model.addVar(vtype="B")

# minimal pref score among all people, that we want to maximize
s = model.addVar(vtype="I", name="min_pref", lb=None)


# setting the objective fucntion
model.setObjective(s, "maximize")

# adding the constraints
for p in people_set:
    # sum of prefs
    sum_of_prefs = quicksum(prefs.loc[p, t] * variables[(p, b, t)]
                            for b in boats_for_person[p]
                            for t in times_avail_for_person[p])
    # s = min_{p} sum of prefs
    model.addCons(s <= sum_of_prefs)

    # each person trains the number of times they asked
    nb_trainings = quicksum(variables[(p, b, t)]
                            for b in boats_for_person[p]
                            for t in times_avail_for_person[p])
    model.addCons(nb_trainings == asked_nb_trainings[p])

    # each person can only row one boat at a time
    for t in times_avail_for_person[p]:
        nb_boats = quicksum(variables[(p, b, t)]
                            for b in boats_for_person[p])
        model.addCons(nb_boats <= 1)

# for each boat and time, no more than one person
for b in set(boats[boats['skill'].isin([0, 1, 2])].index):
    for t in time_set:
        nb_people_on_boat_time = quicksum(variables[(p, b, t)]
                                          for p in people_for_boat_time[(b, t)])
        model.addCons(nb_people_on_boat_time <= 1)

# optimize
model.optimize()


status = model.getStatus()
if status == 'optimal':
    # generate resulting CSV file with boat allocation
    levels = [['M', 'T', 'W', 'Th', 'F', 'S'], ['am1', 'am2', 'pm']]
    cols = pd.MultiIndex.from_product(levels, names=['day', 'time'])
    result = pd.DataFrame(columns=cols, index=people_set)
    result.index.rename('people', inplace=True)

    for p in people_set:
        for t in times_avail_for_person[p]:
            boat = {b for b in boats_for_person[p] if model.getVal(variables[(p, b, t)]) == 1}
            assert len(boat) <= 1
            if boat:
                result.loc[p, t] = boat.pop()

    result.to_csv('results/comp/results_boat.csv')
    print("Fairness score (0 = optimal) : ", model.getVal(s))

else:
    print(f"Solver finished with status {status}")
