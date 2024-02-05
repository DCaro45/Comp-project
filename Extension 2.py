"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

'''values'''
mass = 1   # setting mass to be 1

ti = 0     # start time
tf = 4     # finish time
div_t = 2  # division of time points (i.e whole numbs, half, third etc))

epsilon = 1.4  # change in delta_xs size from spatial lattice spacing
bins = 1000     # number of bins for histogram

N_cor = 25        # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 6    # number of updates

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

print('nt = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) + ', ' 'N_cor/Update = ' + str(N_cor) + ', ' + 'S1 = ' + str(T) + ', ' + 'N_CF = ' + str(N_CF))

def pot1(x):
    V = 1/2 * x ** 2
    return V

def pot2(x):
    V = - 1/3 * x ** 2 + x ** 5 + 5 * np.cos(x)
    return V

def pot3(x):
    """ a polynomial potential with a minimum and a stationary inflection point"""
    V = 1/2 * x ** 2 + 1/4 * x ** 4 - 1/20 * x ** 5
    return V
def pot4(x):
    V = - x ** 2
    return V

def pot5(x):
    if -5 < x < 5:
        V = x
    else:
        V = 100000
    return V

def pot6(x):
    """a potential with the same form as the higgs potential"""
    u = 2
    l = 1
    V = - 0.5 * u ** 2 * x ** 2 + 0.25 * l ** 2 * x ** 4
    return V

pot = pot5

def actn(x, j, potential):
    """calculating energies"""
    # setting index so that it loops around
    jp = (j-1) % nt
    jn = (j+1) % nt

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

def norm(array):
    """normalisation function"""
    total = sum(array)
    if total > 0:
        normalised = array / total
        return normalised
    else:
        return 0


p_1 = [0 for x in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)

"""Thermalising lattice"""
init = p_1
array = [init]
for i in range(T):
    new_p, counts = Metropolis(array[-1], pot)
    array.append(new_p)

"""generating paths and applying metropolis"""
all_ps = []
t_counts = 0
for j in range(N_CF):
    start_p = init
    for i in range(U):
        new_p, counts = Metropolis(start_p, pot)
        start_p = new_p
        t_counts += counts
    all_ps.append(start_p)

print('prop of changing point = ' + str(t_counts/(nt*U*N_CF)))

"""all points fromn new_ps"""
ln = len(all_ps)
pos = np.zeros([ln * nt])
k = 0
for i in range(ln):
    for j in range(nt):
        pos[k] = all_ps[i][j]
        k += 1


xs = np.linspace(min(pos), max(pos), len(pos))
V = []
for x in xs:
    V.append(pot(x))

fig, ax1 = plt.subplots()

ax1.hist(pos, bins)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlabel('Position')
ax1.set_ylabel('Count', color='black')
ax1.set_title('A Histogram of Points from the Metropolis Algorithm Within a Higgs Type Potential')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(xs, V, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylabel('Potential', color='black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#txt = ('A histogram of points produced by the Metropolis Algorithm
#        within a Higgs type potential'
#        )
#plt.figtext(0.5, 0.2, txt, wrap=True, horizontalalignment='center', fontsize=12)

dir, file = os.path.split(__file__)
fig.savefig(dir + '\\Images\\Hist-Higgs.png')
plt.show()