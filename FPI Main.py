import numpy as np
import matplotlib.pyplot as plt
import math as mt
import random as rdm

'''values'''
mass = 1  # setting mass to be 1

x0 = -2  # lower bound
x1 = 2  # upper bound
ti = 0  # start time
tf = 10  # finish time

n_t = 6         # number of temporal points (i.e whole numbs, half, third etc))
div_x = 10       # division of spacial points
N = int(10**5)  # number of paths seeding metropolis

N_fin = int(10e+3)                    # finishing value on logarithmic scale
nbr = 20                         # number of graphs


'''determinants/shorthands'''
n_x = div_x * (x1 - x0) + 1          # number of spatial points
a = (tf - ti) / (n_t - 1)            # size of time step
b = (x1 - x0) / (n_x - 1)            # size of spacial step
t_points = np.linspace(ti, tf, n_t)  # temporal lattice points
x_points = np.linspace(x0, x1, n_x)  # spacial lattice points

base = (N_fin)**(1/nbr)         # base for logarithmic scale
exp = mt.log(N_fin, base)       # exponent for end value of logarithm range



m = mass  # shorthand for mass
nt = n_t  # shorthand for no.t points
nx = n_x  # shorthand for no.x points
x = x_points  # shorthand for spatial lattice points
t = t_points  # shorthand for temporal lattice points
e = exp

"""Defining functions"""
def path_gen(x_0):
    """path generator"""
    path = np.zeros([nt])
    path[0] = path[nt-1] = x_0
    for i in range(1, nt-1):
        path[i] = rdm.choice(x)
    return path

def pot(xs):
    """simple harmonic oscillator potential"""

    potential = 1/2 * xs**2
    return potential

def actn(path, potential):
    """calculating energies"""

    E_path = 0
    for i in range(0, nt-1):
        KE = m/2*((path[i+1]-path[i])/a)**2
        PE = potential((path[i+1]+path[i])/2)
        E_tot = KE + PE
        E_path += E_tot
    Action = a * E_path
    return Action

def wght(action):
    """calculating weight"""

    weight = np.exp(-action)
    return weight

def prop(potential, samples):
    """calculating propagator"""

    propagator = np.zeros([nx])
    run = samples//nx
    rem = samples%nx
    if run > 0:
        for j, x0 in enumerate(x):
            for i in range(int(run+rem)):
                p = path_gen(x0)
                S = actn(p, potential)
                W = wght(S)
                propagator[j] += W
    else:
        for j, x0 in enumerate(x):
            for i in range(rem):
                p = path_gen(x0)
                S = actn(p, potential)
                W = wght(S)
                propagator[j] += W
    print(run, rem, run * nx, samples)
    return propagator

def pdf(xs):
    """prob density function"""

    prob = (np.exp(-(xs ** 2 / 2)) / np.pi ** (1 / 4)) ** 2
    return prob

def norm(array):
    """normalisation function"""

    total = sum(array)
    if total > 0:
        normalised = array/total
        return normalised
    else:
        return 0

p_1 = path_gen(x[int(len(x)/2)])
print(p_1)
g = prop(pot, 10 * nx)
print(g)

'''Calculating and plotting PDF'''
# values
G = prop(pot, N)
#print(G)
Norm_G = norm(G)
y1 = Norm_G

# graph
plt.figure()
plt.plot(x, y1)
plt.show()


'''repeating propagator for smaller samples'''
# sample size
Ns = np.logspace(start=0, stop=e, base=base, num=nbr)

# values
Gs = np.zeros([nbr, nx])
for j in range(nbr):
    for i in Ns:
        Gs[j] = prop(pot, int(i))


# normalising values
Norm_Gs = np.zeros([nbr, nx])
for i in range(nbr):
    Norm_Gs[i] = norm(Gs[i])
ys = Norm_Gs


'plotting graphs'
As = np.linspace(0.25, 1, nbr)    #Alpha valeus

plt.figure(figsize=[20,8])
plt.subplot(1, 2, 1)
for j in range(0, nbr):
    plt.plot(x, ys[j], alpha = As[j])
plt.xlabel('position')
plt.ylabel('probability density')


'''plot of FPI and standard formulation'''
# calculate potential and analytic pdf'
pdf_A = pdf(x)
pdf_B = norm(pdf_A)
y2 = pdf_B * (max(y1)/max(pdf_B))

xs = np.linspace(-5, 5, 100)
ys = pot(xs)

# plotting graphs
plt.subplot(1, 2, 2)
plt.plot(x  , y1, label='FPI',       color='k')
plt.plot(x  , y2, label='PDF',       color='tab:orange')
plt.plot(xs , ys, label='Potential', color='tab:blue')
plt.legend()
plt.grid()
plt.xlim(-2, 2)
plt.xlabel('position')
plt.ylim(0, max(y1) + 0.1*max(y1))

plt.show()
