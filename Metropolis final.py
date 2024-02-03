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

epsilon = 1.3  # change in delta_xs size from spatial lattice spacing
N_cor = 25        # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 5    # number of updates

bins = 100      # number of bins for histogram

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

print('nt = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) +
      ', ' 'N_cor/Update = ' + str(N_cor) + ', ' + 'S1 = ' + str(T) + ', ' + 'N_CF = ' + str(N_CF))

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

def pdf(x):
    """prob density function"""
    prob = ( np.exp( - x ** 2 / 2 ) / ( np.pi ** (1/4) ) ) ** 2
    return prob

p_1 = [0 for x in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)

"""generating applying metropolis to inital path"""

init = p_1
all_ps = []
t_counts = 0
# applying metropolis to path N_CF times
for j in range(N_CF):
    # initialising starting path
    start_p = init
    # applying metropolis to path N_cor times
    for i in range(U):
        new_p, counts = Metropolis(start_p, pot)
        start_p = new_p
        t_counts += counts

    # adding final path to all_ps
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


xs = np.linspace(-3, 3, len(pos))
PDF = pdf(xs)
V = pot(xs)

counts, bins = np.histogram(pos, bins=bins, density=True)
Norm = max(PDF)/max(counts) * counts

fig, ax1 = plt.subplots()

ax1.stairs(Norm, bins, fill=True, label='Monte Carlo integral')
ax1.tick_params(axis='y', labelcolor='red')
ax1.plot(xs, PDF, color='tab:orange', label='analytic solution')
plt.legend()
ax2 = ax1.twinx()
ax2.plot(xs, V, color='tab:green', label='potential')
ax2.tick_params(axis='y', labelcolor='tab:green')
plt.legend(loc='upper left')

fig.tight_layout()
plt.title("The Probability Density Function of a Particle in a Harmonic Oscillator Potential")
plt.xlabel("Position")
plt.ylabel("Probability Density")
dir, file = os.path.split(__file__)
#fig.savefig(dir + '\\Images\\2Dhist.png')
plt.show()

