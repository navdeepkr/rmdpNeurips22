# rmdpNeurips22
Robust Markov Decision Processes: Efficient Policy Iteration for Robust MDPs

##### Repository contains codes and results for the paper:   Efficient Policy Iteration for Robust MDPs that is present here.

## main.py:  
Run it to compare running time of non-robust MDPs, sa-rectangular L_p robust MDPs and sa-rectangular L_p robust MDPs
for p = [1,2,3,4,5,6,7,8,10,infty].   

## pm.py
It contains the parameters for robust and robust MDPs, for example number of state and action, discount factor, uncertainty radius, 
number of iterations, etc
The kernel and reward are generated randomly

## main.txt
It contains the results (output) of main.py run different parameters

## bellman.py
It contains bellman operators for non-robust, sa/s rectangular L_p robust MDPs
