# Boat Allocator
Automatic allocation of boats while maximizing schedule preferences of rowers (using a Mixed Integer Program), used at the Montreal Rowing Club.

The MIP is solved using [PyScipOpt](https://github.com/SCIP-Interfaces/PySCIPOpt), the python interface of the [SCIP](https://www.scipopt.org/) solver. It is run weekly. A few days in advance, rowers submit their schedule preferences via an online form. These preferences consist of "first choices", that correspond to when a rowers would ideally want to row, and "second choices", which are backup times at which they would be available, in case it's not possible for them to row one of their "first choice".

## Model
![model](details/model.svg "Model details")

## Explainations
More details about the objective function and the constraints:
- (1): The objective function is made of two components. The first one $\lambda s$ maximizes the lowest utility value that a person gets. The second one maximizes the sum of utilities across all people. The coefficient $\lambda$ should be large enough to ensure that the maximum value of $s$ is reached before optimizing the sum of utilities. This is to ensure fairness, i.e. make sure it's not possible that a small set of people receive a schedule with very low utility value, just to maximize the overall utility. The minimal value of this coefficient can be computed, but I've been using $500$ which I am sure is enough for the size of my problem, and does not hit the CPU time in a significant way,
- (2): ensure that $s$ is the minimum utility across people,
- (3): cap the max number of training at the number of trainings asked for each person. We don't want the utility to be increased by assigning more trainings to someone than they asked,
- (4): one person can only row one boat per training (otherwise, would increase utility artificially),
- (5): ensure that we don't assign someone to training that are mutually exclusive (according to $E$),
- (6): each boat can only accommodate one person per training.
