# Quick way to use
Make a `py` script, which contains following contents at the directory where `grumpy_cylp` is installed.
```python
# cd location/of/the/grumpy_cylp/folder
from src.cylpBranchAndBound import RELIABILITY_BRANCHING, HYBRID
from src.cylpBranchAndBound import BranchAndBound
from src.generator import GenerateRandomMIP

# extra branching strategies
MOST_FRACTIONAL = 'Most Fraction'
FIXED_BRANCHING = 'Fixed Branching'
PSEUDOCOST_BRANCHING = 'Pseudocost Branching'
# search strategies
DEPTH_FIRST = 'Depth First'
BEST_FIRST = 'Best First'
BEST_ESTIMATE = 'Best Estimate'

T = BBTree()
CONSTRAINTS, VARIABLES, OBJ, MAT, RHS = GenerateRandomMIP(numVars=30, numCons=20, rand_seed=418, density=0.3)
opt, LB, stat1 = BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                             branch_strategy=RELIABILITY_BRANCHING,
                             search_strategy=DEPTH_FIRST,
                             display_interval=10000,
                             solver='primalSimplex',
			     rel_param=(4, 3, 1 / 6, 5),
                             binary_vars=True,
                             more_return=True
                             )
							 
T = BBTree()
opt2, LB2, stat2 = BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                             branch_strategy=HYBRID,
                             search_strategy=DEPTH_FIRST,
                             display_interval=10000,
                             solver='primalSimplex',
                             binary_vars=True,
                             more_return=True
                             )
							 
							 
							 
T = BBTree()
opt3, LB3,, stat3 = BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                             branch_strategy=PSEUDOCOST_BRANCHING,
                             search_strategy=DEPTH_FIRST,
                             display_interval=10000,
                             solver='primalSimplex',
                             binary_vars=True,
                             more_return=True
                             )
print(RELIABILITY_BRANCHING)
print(stat1)
print(HYBRID)
print(stat2)
print(PSEUDOCOST_BRANCHING)
print(stat3)
```
Or you can simply call `cylpBranchAndBound` as a module by

```bash
cd location/of/the/grumpy_cylp/folder
python src/cylpBranchAndBound
```


Branch strategy can be chosen from `MOST_FRACTIONAL`, `FIXED_BRANCHING`, `PSEUDOCOST_BRANCHING`, `RELIABILITY_BRANCHING`, `RELIABILITY_BRANCHING`, and `HYBRID`.

Search strategy can be chosen from `DEPTH_FIRST`, `BEST_FIRST`, and `BEST_ESTIMATE`.

Solver can be chosen from `dynamic`, which pick primal or dual simplex automatically, `primalSimplex`, and `dualSimplex`.

`rel_param` are for reliability branching only, see comments in `cylpBranchAndBound.py` for detailed explanation on the choices of each values.

If `more_return` is true, then function returns optimizer, optimal objetive values, and statistics that includes time(in ms) for branch and bound, number of nodes, and number LP solved;
otherwise only returns optimizer and optimal objetive values.


