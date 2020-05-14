# Quick way to use
Make a `py` script, which contains following contents at the directory where `grumpy_cylp` is installed.
```python
# cd location/of/the/grumpy_cylp/folder
from src.cylpBranchAndBound import RELIABILITY_BRANCHING, HYBRID
from src.cylpBranchAndBound import BranchAndBound
from src.generator import GenerateRandomMIP
T = BBTree()
T.set_display_mode('xdot')
CONSTRAINTS, VARIABLES, OBJ, MAT, RHS = GenerateRandomMIP(numVars=30, numCons=20, rand_seed=418, density=0.3)
_, _, stat1 = BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                             branch_strategy=RELIABILITY_BRANCHING,
                             search_strategy=DEPTH_FIRST,
                             display_interval=10000,
                             solver='primalSimplex',
                             binary_vars=True,
                             more_return=True
                             )
T = BBTree()
T.set_display_mode('xdot')
_, _, stat2 = BranchAndBound(T, CONSTRAINTS, VARIABLES, OBJ, MAT, RHS,
                             branch_strategy=PSEUDOCOST_BRANCHING,
                             search_strategy=DEPTH_FIRST,
                             display_interval=10000,
                             solver='primalSimplex',
                             binary_vars=True,
                             more_return=True
                             )
print(RELIABILITY_BRANCHING)
print(stat1)
print(PSEUDOCOST_BRANCHING)
print(stat2)
```
Or you can simply call `cylpBranchAndBound` as a module by

```bash
cd location/of/the/grumpy_cylp/folder
python src/cylpBranchAndBound
```
