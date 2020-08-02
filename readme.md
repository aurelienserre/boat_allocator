# Boat Allocator
Automatic allocation of boats while maximizing schedule preferences of rowers (using a Mixed Integer Program), used at the Montreal Rowing Club.

The MIP is solved using [PyScipOpt](https://github.com/SCIP-Interfaces/PySCIPOpt), the python interface of the [SCIP](https://www.scipopt.org/) solver. It is run weekly. A few days in advance, rowers submit their schedule preferences via an online form. These preferences consist of "first choices", that correspond to when a rowers would ideally want to row, and "second choices", which are backup times at which they would be available, in case it's not possible for them to row one of their "first choice".

## Model
![model](details/model.svg "Model details")
