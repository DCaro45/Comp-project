import numpy as np
import math as mt
import random as rdm

mass = 1

x0 = -2       #lower bound
x1 = 2        #upper bound
ti = 0        #start time
tf = 5        #finish time

n_t = 7                         #Number of time points
divs_x = 2                      #Division of space points (i.e whole numbs, half, third etc)
n_x = divs_x * (x1 - x0) + 1    #number of spatial points

'sample sizes'
N= int(10e+1)                   #number of samples for prop calc

N_fin = 10e+4                   #finishing point logarithmic scale
base = 2                        #base for logarithmic scale
nbr = 10                        #number of graphs


'''determinants/shorthands'''
m = mass                       #shorthand for mass

a = (tf-ti)/(n_t-1)                  #size of time step
t_points = np.linspace(ti, tf, n_t)  #time lattice points
x_points = np.linspace(x0, x1, n_x)  #spacial lattice points

n = n_x                          #shorthand for no.x points
x = x_points                     #shorthand for lattice points

exp = mt.log(N_fin, base)       #exponent for end value of logarithm range


def path_gen(xs, x0):
    """path generator"""

    lgt = len(xs)
    path = np.zeros([lgt])
    path[0]=path[lgt-1]=x0
    for i in range(1,lgt-1):
        path[i]=rdm.choice(xs)
    return path

'''values'''
mass = 1

x0 = -2       #lower bound
x1 = 2        #upper bound
ti = 0        #start time
tf = 5        #finish time

n_t = 7                         #Number of time points
divs_x = 2                      #Division of space points (i.e whole numbs, half, third etc)
n_x = divs_x * (x1 - x0) + 1    #number of spatial points

_size_ = 10                    #how much smaller the delta_xs shouls be from lattice spacing

'sample sizes'
N= int(10e+1)                   #number of samples for prop calc

'''determinants/shorthands'''
m = mass                       #shorthand for mass

a = (tf-ti)/(n_t-1)                  #size of time step
b = (x1-x0)/(n_x-1)                  #size of spacial step
t_points = np.linspace(ti, tf, n_t)  #time lattice points
x_points = np.linspace(x0, x1, n_x)  #spacial lattice points

n = n_x                          #shorthand for no.x points
x = x_points                     #shorthand for lattice points


def path_gen(xs, x0):
    """path generator"""

    lgt = len(xs)
    path = np.zeros([lgt])
    path[0]=path[lgt-1]=x0
    for i in range(1,lgt-1):
        path[i]=rdm.choice(xs)
    return path

def pot(x):
    """simple harmonic oscillator potential"""

    V = 1/2*(x)**2
    return V

def Actn(path, potential):
    """calculating energies"""

    E_path = [0]
    for i in range(0, n-1):
        KE = m/2*((path[i+1]-path[i])/a)**2
        PE = potential((path[i+1]+path[i])/2)
        E_tot = KE + PE
        E_path+=  E_tot
    Action = a * E_path
    return Action

def Wght(action):
    """calculating weight"""

    weight = np.exp(-action)
    return weight


def Metropolis(size, path, point, potential, energy):
    """creating the metropolis algorithm"""
    dev = b/size
    d_x = rdm.uniform(-dev, dev)

    #old E
    old_p = path
    E_old = energy(old_p, potential)
    print(E_old)

    #new E
    path[point] = path[point] + d_x
    new_p = path
    E_new = energy(new_p, potential)
    print(E_new)

    #delta S
    d_S = a * (E_new - E_old)

    e_path = []
    #Conditional statement
    if d_S < 0:
        e_path = new_p
    elif d_S > 0:
        r = rdm.random()
        weight = Wght(d_S)
        if weight > r:
            e_path = new_p
    else:
        e_path = old_p
    return e_path

def init(samples):
    """initialises a load of empty paths that start and end at the same point ina lattice"""


def prop(points, potential, path, action, samples):
    """calculating propagator"""
    'cal variables'
    l_size = len(points)
    indx = np.linspace(0,l_size, l_size+1)
    size = _size_
    N_cor = 1 / (b) ** 2
    N_up = 10*N_cor
    lgth = int(samples/(N_up))

    #initialising lattice
    paths = np.zeros([samples, l_size])

    #thermalisation
    therm_p = np.zeros([lgth, l_size])
    for i in range(lgth):
        point = paths[i * N_up][rdm.choice[indx]]
        therm_p[i] = Metropolis(size, paths[i * N_up], point, pot, action)

    #update
    fin_ps = np.zeros([lgth, l_size])
    for i in range(lgth):
        point = therm_p[i][rdm.choice[indx]]
        therm_p[i] = Metropolis(size, therm_p[i], point, pot, action)

    weight = np.zeros([lgth])
    for i in range(lght):
        weight[i] = Wght(Actn(fin_ps[i], potential))

    G = np.zeros([l_size])
    run = int(samples/l_size)
    for x0 in points:
        for i in range(0, run):
            p = path(points, x0)
            E = energy(p, potential)
            W = Wght(E)
            indx = np.where(points == x0)
            G[indx] += W
    return G

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


'''Plotting PDF'''
#values
G = prop(x, pot, path_gen, Actn, N)
Norm_G = norm(G)
y1 = Norm_G

#Graph
plt.figure()
plt.plot(x, y1)
plt.show()

