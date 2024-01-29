"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

'''values'''
mass = 1   # setting mass to be 1

ti = 0     # start time
tf = 3     # finish time
div_t = 3   # division of time points (i.e whole numbs, half, third etc))

epsilon = 1.6  # change in delta_xs size from spatial lattice spacing
N_cor = 25        # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 5    # number of updates

'''determinants/shorthands'''
n_tp = div_t * (tf - ti) + 1          # number of temporal points
n_tl = div_t * (tf - ti)              # number of temporal links
a = (tf - ti) / n_tl                  # size of time step
t_points = np.linspace(ti, tf, n_tp)  # temporal lattice points

Therm = 10 * N_cor    # number of sweeps through path set
Update = N_cor        # not necessary right now

m = mass           # shorthand for mass
nt = n_tp          # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)
U = int(Update)    # shorthand for sweeps 2 (and integer data type)

print('# temporal points = ' + str(nt) + ', ' + 'size of time step = ' + str(a) + ', ' + 'temporal points = ' + str(t) + ', ' + 'epsilon = ' + str(e) +
      ', ' '# updates (N_cor) = ' + str(N_cor) + ', ' + 'Therm sweeps (T) = ' + str(T) + ', ' + '# paths (N_CF) = ' + str(N_CF))

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
        xP = x + dx
        eval_p = path.copy()
        eval_p[j] = xP

        # calculating actions
        S1 = actn(path, j, potential)
        S2 = actn(eval_p, j, potential)
        ds = S2 - S1

        # applying metropolis logic
        r = rdm.random()
        W = np.exp(-ds)
        if ds < 0 or W > r:
            path = eval_p
            count += 1

    return path, count

def compute_G1(x,n):
    g = 0
    for j in range(nt):
        jn = (j+n)%nt
        g += x[j] * x[jn]
    return g/nt

def compute_G2(x,n):
    g = 0
    for j in range(nt):
        jn = (j+n)%nt
        g += x[j] ** 3 * x[jn] ** 3
    return g/nt

def delta_E(prop):
    dE = []
    for n in range(nt):
        k = (n+1)%nt
        dE_i = np.log(prop[n]/prop[k]) / a
        dE.append(dE_i)
    return dE

compute_G = compute_G2

p_1 = [0 for x in range(nt)]
p_2 = [np.random.uniform(-4, 4) for x in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)

G = compute_G(p_2, 0)
print(G)

"""Thermalising lattice"""
init = p_1
array = [init]
for i in range(T):
    x, count = Metropolis(array[-1], pot)
    array.append(x)

"""generating array of G values"""
G = np.zeros([N_CF, nt])
count = 0
for alpha in range(N_CF):
    start_x = array[-1]
    for j in range(U):
        new_x, c = Metropolis(start_x, pot)
        start_x = new_x
        count += c
    for n in range(nt):
        G[alpha][n] = compute_G(x,n)

Av_G = []
for n in range(nt):
    avg_G = 0
    for alpha in range(N_CF):
        avg_G += G[alpha][n]
    avg_G = avg_G/N_CF
    Av_G.append(avg_G)
print('G(%d) = %g' % (n, avg_G) + ', ' + str(count/(nt*U*N_CF)))
print(Av_G)
dEs = delta_E(Av_G)
dE = [1 for t in range(nt)]

plt.figure(figsize=[8, 4])
plt.plot(t, dE, linestyle='--', color='black')
plt.scatter(t, dEs)
plt.xlabel('t')
plt.ylabel('$\Delta$E(t)')
plt.show()
dir, file = os.path.split(__file__)
#fig.savefig(dir + '\\Images\\3Dhist.png')




