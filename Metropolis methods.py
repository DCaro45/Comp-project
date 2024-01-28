"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt

'''values'''
mass = 1  # setting mass to be 1

ti = 0   # start time
tf = 4   # finish time

div_t = 2  # division of time points (i.e whole numbs, half, third etc))

epsilon = 0.5  # change in delta_xs size from spatial lattice spacing
bins = 100  # number of bins for histogram

'''determinants/shorthands'''
n_t = div_t * (tf - ti) + 1  # number of spatial points
n_t = 7
a = (tf - ti) / (n_t - 1)  # size of time step
t_points = np.linspace(ti, tf, n_t)  # temporal lattice points

N_cor = 10            # number of paths to be skipped path set (due to correlation)
Sweeps1 = 10 * N_cor  # number of sweeps through path set
Sweeps2 = N_cor       # not necessary right now
N_CF = 10 ** 2           # number of updates

m = mass  # shorthand for mass
nt = n_t  # shorthand for no.t points
t = t_points  # shorthand for temporal lattice points
e = epsilon
S1 = int(Sweeps1)  # shorthand for sweeps 1 (and integer data type)
S2 = int(Sweeps2)  # shorthand for sweeps 2 (and integer data type)


def pot(x):
    """simple harmonic oscillator potential"""
    V = 0.5 * x ** 2
    return V

def actn(x, j, potential):
    """calculating energies"""
    jp = (j-1) % nt
    jn = (j+1) % nt

    KE = m * x[j] * (x[j] - x[jp] - x[jn]) / (a ** 2)
    PE = potential(x[j])
    E_tot = KE + PE
    Action = a * E_tot

    return Action

def Metropolis(path, potential):
    """creating the metropolis algorithm"""
    count = 0
    dS = []

    for j, x in enumerate(path):
        eval_p = path.copy()

        dx = rdm.uniform(-e, e)
        xP = x + 0.1
        eval_p[j] = xP


        S1 = actn(path, j, potential)
        S2 = actn(eval_p, j, potential)
        ds = S2 - S1

        dS.append(ds)
        r = rdm.random()
        W = np.exp(-ds)
        if ds > 0 and W < 0.5:
            pass
        else:
            path = eval_p
            count += 1

    return path, dS

def V(m, x):
    return 0.5 * m * x ** 2

def delta_action(xp, x1, x0, xu, V):
    s1 = 0.5 * m * (1 / a) * ((x1 - xu) ** 2 + (x0 - xu) ** 2) + a * V(m, xu)
    s2 = 0.5 * m * (1 / a) * ((x1 - xp) ** 2 + (x0 - xp) ** 2) + a * V(m, xp)
    return s2 - s1

def sweep(path, N, V):
    #    this is a single sweep, path is N long
    N = nt

    dS = []

    for i in range(N):

        x_perturbed = path[i] + 0.1

        s_delta = delta_action(x_perturbed, path[(i + 1) % N], path[(i - 1) % N], path[i],
                               V)  # Calculates the change in action
        dS.append(s_delta)

        # Determines whether to keep the change
        if (s_delta < 0):
            path[i] = x_perturbed
        elif (0.5 < np.exp(-s_delta)):
            path[i] = x_perturbed

    return path, dS


def energy(mass, x, j):
    N = nt
    eps = a
    jp = (j+1)%N
    jm = (j-1)%N

    pe = V(mass, x[j])
    ke = x[j]*(x[j]-x[jp]-x[jm]) / (eps**2)
    S = a * (pe + ke)

    return S

def therm_sweep(starting_points, acceptance_ratio):

    dS = []
    for i in range(len(starting_points)):
        initial_points = starting_points.copy()
        initial_energy = energy(1, initial_points, i)

        perturbed_points = starting_points.copy()
        perturbed_points[i] += 0.1
        perturbed_energy = energy(1, perturbed_points, i)

        energy_diff = perturbed_energy - initial_energy

        dS.append(energy_diff)

        eps = a
        if energy_diff < 0 or 0.5 < np.exp(-eps * energy_diff):
            acceptance_ratio.append(1)
            starting_points[i] = perturbed_points[i]
        else:
            acceptance_ratio.append(0)

    return starting_points, dS


print('nt = ' + str(nt) + ',', 't = ' + str(t) + ',', 'S1 = ' + str(S1) + ',', 'N_cor = ' + str(N_cor))


accept = []

p_1 = [np.random.uniform(0,7) for x in range(nt)]
p_2 = p_1.copy()
p_3 = p_1.copy()
p1, dS_1 = Metropolis(p_1, pot)
p2, dS_2 = sweep(p_2, nt, V)
p3,dS_3 = therm_sweep(p_3, accept)


print(p1, p2, p3)
print(dS_1, dS_2, dS_3)

for i in range(len(p1)):
    if float('%.2g' % p1[i]) == float('%.2g' % p2[i]) == float('%.2g' % p3[i]):
        print(True)
    else:
        print(False)

for i in range(len(dS_1)):
    if float('%.2g' % dS_1[i]) == float('%.2g' % dS_2[i]) == float('%.2g' % dS_3[i]):
        print(True)
    else:
        print(False)


def path_check(path_old, path_new):
    return

def pdf(x):
    """prob density function"""
    prob = (np.exp(-(x ** 2 / 2)) / np.pi ** (1 / 4)) ** 2
    return prob


def norm(array):
    """normalisation function"""

    total = sum(array)
    if total > 0:
        normalised = array / total
        return normalised
    else:
        return 0



'''

"""Thermalising lattice"""
init = p_1
array = [init]
for i in range(S1):
    new_p, counts = Metropolis(array[-1], pot)
    array.append(new_p)

"""generating paths and applying metropolis"""
all_ps = []
t_counts = 0
for j in range(N_CF):
    start_p = array[-1]
    for i in range(S2):
        new_p, counts = Metropolis(start_p, pot)
        start_p = new_p
        t_counts += counts
    all_ps.append(start_p)

#print(all_ps)
print('prop of changing point = ' + str(t_counts/(nt*N_CF*S2)))


"""points from new_ps skipping every N_cor'th one"""

pos_1 = np.zeros([int(len(all_ps)/N_cor) * nt])
print(N_cor, len(all_ps), len(all_ps) * nt, len(pos_1))
m = 0
for i in range(int(len(all_ps)/N_cor)):
    for j in range(nt):
        pos_1[m] = all_ps[int(i*N_cor)][j]
        m += 1
#print(pos_1)


"""all points fromn new_ps"""
ln = len(all_ps)
pos_2 = np.zeros([ln * nt])
m = 0
for i in range(ln):
    for j in range(nt):
        pos_2[m] = all_ps[i][j]
        m += 1
#print(pos_2)

xs = np.linspace(min(pos_1), max(pos_1), len(pos_1))
pdf_A = pdf(xs)
Norm = norm(pdf_A)

counts, bins = np.histogram(pos_1, bins = bins)
y2 = (max(counts)/max(Norm))*norm(pdf_A)

plt.hist(all_ps, bins=bins, label ='M4', histtype="step")
#plt.plot(xs , y2, label = 'PDF', color = 'tab:orange' )
plt.legend()
plt.show()

xs = np.linspace(min(pos_2), max(pos_2), len(pos_2))
pdf_A = pdf(xs)
Norm = norm(pdf_A)

counts, bins = np.histogram(pos_2, bins = bins)
y2 = (max(counts)/max(Norm))*norm(pdf_A)


plt.hist(pos_2, bins=bins, label = 'M4')
plt.plot(xs , y2, label = 'PDF', color = 'tab:orange' )
plt.legend()
plt.show()

'''