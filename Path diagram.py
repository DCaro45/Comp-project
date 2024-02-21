"""showing how the path is creating for brute force and metropolis algorithms"""

import numpy as np
import matplotlib.pyplot as plt
import random as rdm
import os

dir, file = os.path.split(__file__)


##values
number = 2  #Division of ticks (i.e whole numbs, half, third etc)
x1 = 2      #upper bound
x0 = -x1    #lower bound
ti = 0      #start time
tf = 5      #finish time


##determinants

n = number * (x1 - x0) + 1     #number of steps
n = n
a = (tf-ti)/(n-1)                  #time step
points = np.linspace(x0,x1,n)  #spacial lattice points

x = points       #shorthand for lattice points

'''values'''
mass = 1   # setting mass to be 1

t_i = 0     # start time
t_f = 4     # finish time
div = 2     # division of time points (i.e whole numbs, half, third etc))

epsilon = 10   # change in delta_xs size from spatial lattice spacing
bins = 100     # number of bins for histogram

N_cor = 20        # number of paths to be skipped path set (due to correlation)
Therm = 20 * N_cor    # number of sweeps through path set

'''determinants/shorthands'''
n_tp = int( div * (t_f - t_i) + 1 )         # number of temporal points
n_tl = int( div * (t_f - t_i) )             # number of temporal links
a = (t_f - t_i) / n_tl                  # size of time step
t_points = np.linspace(t_i, t_f, n_tp)  # temporal lattice points


m = mass           # shorthand for mass
nt = n_tp           # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
U = int(N_cor)     # shorthand for sweeps 2 (and integer data type)
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)



def pot(x):
    """simple harmonic oscillator potential"""
    V = 1/2 * x ** 2
    return V


def actn(x, j, potential):
    """calculating energies"""
    # setting index so that it loops around
    N = len(x)
    jp = (j-1) % N
    jn = (j+1) % N

    # calculating energies ... strange???
    KE = m * x[j] * (x[j] - x[jp] - x[jn]) / (a ** 2)
    PE = potential(x[j])
    E_tot = KE + PE
    Action = a * E_tot

    return Action

def Metropolis(path, potential):
    """creating the metropolis algorithm"""
    # keeping count of number of changes
    count = 0

    for j, x in enumerate(path):
        # creating a perturbed path from initial path
        dx = rdm.uniform(-e, e)

        eval_p = path.copy()
        eval_p[j] = x + dx

        # calculating actions
        S1 = actn(path, j, potential)
        S2 = actn(eval_p, j, potential)
        dS = S2 - S1

        # applying metropolis logic
        r = rdm.random()
        W = np.exp(-dS)
        if dS < 0 or W > r:
            path = eval_p
            count += 1

    return path, count


"""Initialising paths and trialing metropolis"""

p_1 = [0 for x in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)


"""Thermalising lattice"""

init = p_1
for i in range(T):
    new_p, counts = Metropolis(init, pot)
    init = new_p
    plt.plot(init,t_points, alpha=0.2)
plt.show()

##path

def path_gen(xs, n, x0):
    path = np.zeros([n])
    path[0]=path[n-1] = x0
    for i in range(1,n-1):
        path[i]=rdm.choice(xs)
    return path

p_1 = path_gen(x, n, x[2])


###graph

samples = 5

#time points
ts = np.linspace(ti, tf, n)

plt.figure(figsize = [3, 4])

for i in range(0,samples):
    p = path_gen(x, n, x[4])
    plt.plot(p, ts)

plt.grid(axis = 'x')
plt.show()