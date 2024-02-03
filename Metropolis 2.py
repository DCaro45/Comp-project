import numpy as np
import math as mt
import random as rdm
import matplotlib.pyplot as plt

'''values'''
mass = 1

x0 = -5       #lower bound
x1 = 5        #upper bound
ti = 0        #start time
tf = 15        #finish time

n_t = 9                         #Number of time points
divs_x = 2                      #Division of space points (i.e whole numbs, half, third etc)

_size_ = 5                    #how much smaller the delta_xs shouls be from lattice spacing
N= 10e+2                      #number of samples for prop calc

'''determinants/shorthands'''
m = mass                       #shorthand for mass

n_x = divs_x * (x1 - x0) + 1    #number of spatial points
a = (tf-ti)/(n_t-1)                  #size of time step
b = (x1-x0)/(n_x-1)                  #size of spacial step
t_points = np.linspace(ti, tf, n_t)  #time lattice points
x_points = np.linspace(x0, x1, n_x)  #spacial lattice points

n = n_x                          #shorthand for no.x points
x = x_points                     #shorthand for lattice points

N = int(N)
Sweeps = 10e+3
N2 = int(Sweeps)
bins = 40


def path_gen2(xs, N):
    """path generator"""
    #n is the length of the spatial lattive array
    N = int(N)
    div = int(N/n)
    paths = np.zeros([N, n])
    indx = 0
    for x0 in xs:
        for m in range(div):
            for i in range(n):
                paths[indx][i] = x0
            indx +=1
    return paths

def pot(x):
    """simple harmonic oscillator potential"""

    V = 1/2*(x)**2
    return V

def actn(path, potential):
    """calculating energies"""

    E_path = [0]
    for i in range(0, n-1):
        KE = m/2*((path[i+1]-path[i])/a)**2
        PE = potential((path[i+1]+path[i])/2)
        E_tot = KE + PE
        E_path+=  E_tot
    Action = a * E_path
    return Action

def wght(action):
    """calculating weight"""
    weight = np.exp(-action)
    return weight

def pdf(x):
    """prob density function"""

    prob = (np.exp(-(x ** 2 / 2)) / np.pi ** (1 / 4)) ** 2
    return prob

def norm(array):
    """normalisation function"""

    total = sum(array)
    if total > 0:
        normalised = array/total
        return normalised
    else:
        return 0

def Metropolis(path, potential):
    """creating the metropolis algorithm"""
    epsilon = b/_size_
    e = epsilon
    dx = rdm.uniform(-e, e)
    indx = rdm.randrange(0, n)
    #print(indx)

    #old E
    old_p, new_p = path, path
    E_old = actn(old_p, potential)

    #new E
    new_p = np.zeros([n])
    for i, x in enumerate(path):
        if i == indx:
            new_p[i] = x + dx
        else:
            new_p[i] = x
    E_new = actn(new_p, potential)

    #delta S
    dS = a * (E_new - E_old)

    #conditional statement
    #cont = []
    eval_p = []
    'make a dS cut off'
    if dS < 0:
        eval_p = new_p
        #cont = True
    elif dS > 0:
        r = rdm.random()
        W = wght(dS)
        if W > r:
            eval_p = new_p
            #cont = True
        else:
            eval_p = old_p
            #cont = False
    return eval_p #, cont


old_ps = path_gen2(x, N)
new_ps = old_ps
j = 0
for i in range(N2):
    for j in range(N):
        new_ps[j] = Metropolis(old_ps[j], pot)
#print(new_ps)

pos = np.zeros([N * n])
m = 0
for i in range(N):
    for j in range(n):
        pos[m] = new_ps[i][j]
        m += 1

xs = np.linspace(min(pos), max(pos), len(pos))
pdf_A = pdf(xs)
Norm = norm(pdf_A)

counts, bins = np.histogram(pos, bins = bins)
y2 = (max(counts)/max(Norm))*norm(pdf_A)

plt.hist(pos, bins=bins)
plt.plot(xs , y2, label = 'PDF', color = 'tab:orange' )
plt.show()