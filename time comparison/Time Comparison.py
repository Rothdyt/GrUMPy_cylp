# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:14:56 2020

@author: Muqing Zheng
"""

import numpy as np
import math,sys,os,time
from random import seed
from random import randint
import matplotlib.pyplot as plt
import pandas as pd

import coinor.grumpy
from coinor.grumpy import GenerateRandomMIP
from coinor.grumpy import BBTree
from coinor.grumpy import MOST_FRACTIONAL, FIXED_BRANCHING, PSEUDOCOST_BRANCHING
from coinor.grumpy import DEPTH_FIRST, BEST_FIRST, BEST_ESTIMATE
from coinor.grumpy import INFINITY

project_dir = '../'
sys.path.append(project_dir)

from src.cylpBranchAndBound import RELIABILITY_BRANCHING, HYBRID
from src.cylpBranchAndBound import BranchAndBound


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
    
def identity(n):
    m=[[0 for x in range(n)] for y in range(n)]
    for i in range(0,n):
        m[i][i] = 1
    return m

def negIdentity(n):
    m=[[0 for x in range(n)] for y in range(n)]
    for i in range(0,n):
        m[i][i] = -1
    return m


blockPrint()
# input Parameters 
M = 30  # Number of Problems
seed(1020)



##################################################
#branch= [MOST_FRACTIONAL, FIXED_BRANCHING, PSEUDOCOST_BRANCHING,RELIABILITY_BRANCHING,HYBRID]
package = ['GrUMPy','CyLP']
branch= [PSEUDOCOST_BRANCHING]
search = [DEPTH_FIRST, BEST_FIRST, BEST_ESTIMATE]
costs_time = {p + ' - '+ i + ' - ' + j:np.array([]) for p in package for i in branch for j in search}


# Solve problems and record tree size of costs
for k in range(M):
    # Problem Size will be random
    numVars = randint(10,40)
    numCons = randint(int(numVars/4),int(2 * numVars/3))
    rand_seed = randint(1,2000)
    #prob_data = np.append(prob_data,(numVars,numCons,rand_seed))
    CONSTRAINTS, VARIABLES, OBJ, MAT, RHS = GenerateRandomMIP(numVars=numVars , numCons=numCons,rand_seed= rand_seed)
        
    for i in branch:
        for j in search:
            T = BBTree()
            start = time.time()
            opt, LB = BranchAndBound(
                T, CONSTRAINTS, VARIABLES, OBJ,MAT, RHS,branch_strategy=i,search_strategy=j)
            end = time.time()
            tol_time = end-start
            if LB>-INFINITY:
                costs_time['CyLP' + ' - ' +  i + ' - ' + j] = np.append(costs_time['CyLP' + ' - ' +  i + ' - ' + j],tol_time)
            else:
                costs_time['CyLP' + ' - ' +  i + ' - ' + j] = np.append(costs_time['CyLP' + ' - ' +  i + ' - ' + j],INFINITY)
                
            T = BBTree()
            start2 = time.time()
            opt2, LB2 = coinor.grumpy.BranchAndBound(
                T, CONSTRAINTS, VARIABLES, OBJ,MAT, RHS,branch_strategy=i,search_strategy=j)
            end2 = time.time()
            tol_time2 = end2-start2
            if LB>-INFINITY:
                costs_time['GrUMPy' + ' - ' +  i + ' - ' + j] = np.append(costs_time['GrUMPy' + ' - ' +  i + ' - ' + j],tol_time2)
            else:
                costs_time['GrUMPy' + ' - ' +  i + ' - ' + j] = np.append(costs_time['GrUMPy' + ' - ' +  i + ' - ' + j],INFINITY)
                
                
def performance_profile(costs,name):
    # Do some computations
    # Calculate Minimum Cost of Each Problem
    min_costs = np.ones(M)*math.inf
    for p in package:
        for i in branch:
            for j in search:
                for k in range(M):
                     if costs[p + ' - '+ i + ' - ' + j][k]<min_costs[k]:
                        min_costs[k] = costs[p + ' - '+ i + ' - ' + j][k]

    # Calculate Ratio of Each Problem  with Each Method              
    ratios = costs
    for p in package:
        for i in branch:
            for j in search:
                for k in range(M):
                    ratios[p + ' - '+ i + ' - ' + j][k] = costs[p + ' - '+ i + ' - ' + j][k]/min_costs[k]

    # Efficients
    effs = np.zeros(len(package)*len(branch) * len(search))
    ind = 0
    for rk in ratios.keys():
        effs[ind] = np.sum(ratios[rk]<=1)/M
        ind += 1

    # Robustness    
    rmax = 0
    for rk in ratios.keys():
        if np.max(ratios[rk]) >rmax:
            rmax = np.max(ratios[rk])

    robs = np.zeros(len(package)*len(branch) * len(search))
    ind = 0
    for rk in ratios.keys():
        robs[ind] = np.sum(ratios[rk]<=rmax)/M
        ind += 1

    # Print a table
    d = {'Method':list(ratios.keys()),'Efficiency':effs,'Robustness':robs}
    df = pd.DataFrame(data=d)
    with open(name + '.tex', 'w') as tf:
         tf.write(df.to_latex(index=False))
            
    # Do plot
    all_rs = np.array([])
    for rk in ratios.keys():
        all_rs = np.append(all_rs,ratios[rk])

    # x-axis
    t = np.sort(np.unique(all_rs))
    # for rk in ratios.keys():
    #     plt.plot(t,[np.sum(ratios[rk]<=t[i])/M for i in range(len(t)) ],label = rk)

    # plt.xlabel('Performance Ratio')
    # plt.ylabel('Percents of Problem Solved')
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.savefig(name + ' ' + 'Performance_Profile_All')
    # plt.show()
    
    for i in branch:
        for rk in ratios.keys():
            if i in rk:
                plt.plot(t,[np.sum(ratios[rk]<=t[i])/M for i in range(len(t)) ],label = rk)
        plt.xlabel('Performance Ratio')
        plt.ylabel('Percents of Problem Solved')
        plt.legend( loc='lower right')
        plt.savefig(name + ' ' + i)
        plt.show()

    for j in search:
        for rk in ratios.keys():
            if j in rk:
                plt.plot(t,[np.sum(ratios[rk]<=t[i])/M for i in range(len(t)) ],label = rk)
        plt.xlabel('Performance Ratio')
        plt.ylabel('Percents of Problem Solved')
        plt.legend(loc='lower right')
        plt.savefig(name + ' ' + j)
        plt.show()
        
        
        
if __name__ == '__main__':
    enablePrint()
    performance_profile(costs_time, 'Pack Time')

