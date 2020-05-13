'''
File: generator.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-05-13 12:53
Last Modified: 2020-05-13 12:53
--------------------------------------------
Description:
'''


def GenerateRandomMIP(numVars=40, numCons=20, density=0.2,
                      maxObjCoeff=10, maxConsCoeff=10,
                      tightness=2, rand_seed=2, layout='dot'):
    random.seed(rand_seed)
    CONSTRAINTS = ["C" + str(i) for i in range(numCons)]
    if layout == 'dot2tex':
        VARIABLES = ["x_{" + str(i) + "}" for i in range(numVars)]
    else:
        VARIABLES = ["x" + str(i) for i in range(numVars)]
    OBJ = dict((i, random.randint(1, maxObjCoeff)) for i in VARIABLES)
    MAT = dict((i, [random.randint(1, maxConsCoeff)
                    if random.random() <= density else 0
                    for j in CONSTRAINTS]) for i in VARIABLES)
    RHS = [random.randint(int(numVars * density * maxConsCoeff / tightness),
                          int(numVars * density * maxConsCoeff / 1.5))
           for i in CONSTRAINTS]
    return CONSTRAINTS, VARIABLES, OBJ, MAT, RHS
