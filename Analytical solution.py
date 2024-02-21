"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

'''values'''
mass = 1   # setting mass to be 1
m = mass

ti = 0     # start time
tf = 4     # finish time
div_t = 1  # division of time points (i.e whole numbs, half, third etc))

n_tp = div_t * (tf - ti) + 1          # number of temporal points
n_tl = div_t * (tf - ti)              # number of temporal links
a = (tf - ti) / n_tl                  # size of time step
t_points = np.linspace(ti, tf, int(n_tp))  # temporal lattice points


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

def pdf(x):
    """prob density function"""
    prob = np.exp(- x ** 2) / (np.pi ** (1/2))
    return prob


def actn(x, potential):
    """calculating energies"""
    # calculating energies ... strange???
    KE = 1/2 * m * (x/a) ** 2
    #KE = 0
    PE = potential(x)
    E_tot = KE + PE
    Action = a * E_tot
    return Action

pot = pot1

xs = np.linspace(-5, 5, 1000)
y = np.zeros([len(xs)])
for i, x in enumerate(xs):
    y[i] = np.exp(-actn(x, pot))

V = pot(xs)
P = pdf(xs)
plt.plot(xs, max(P) * y)
plt.show()


