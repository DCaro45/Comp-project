import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpt
import random as rdm

##values
mass = 1
number = 1  #Division of ticks (i.e whole numbs, half, third etc)
x1 = 2      #upper bound
x0 = -x1    #lower bound
ti = 0      #start time
tf = 5      #finish time


##determinants

n = number * (x1 - x0) + 1     #number of steps
n = n
a = (tf-ti)/n                  #time step
points = np.linspace(x0,x1,n)  #spacial lattice points

x = points       #shorthand for lattice points
m = mass         #shorthand for mass


##path

def path_gen(xs, n):
    'path generator'

    path = np.zeros([n])
    path[0]=path[n-1]=rdm.choice(xs)
    for i in range(1,n-1):
        path[i]=rdm.choice(xs)
    return path

p_1 = path_gen(x, n)


##potential energies

def pot(x):
    'simple harmonic oscillator potential'
    V = 1/2*(x)**2
    return V


##energy
def E(path, potential):
    'calculating energies'
    E_path = [0]
    for i in range(0, n-1):
        KE = m/2*((path[i+1]-path[i])/a)**2
        PE = potential((path[i+1]+path[i])/2)
        E_tot = KE + PE
        E_path+=  E_tot
    return E_path

e_1 = E(p_1, pot)



##weighting
def W(energy):
    'calculating weight'
    weight = np.exp(-a*energy)
    return weight


###propogator

N= int(10e+3) #number of samples

def prop(points, potential, path, energy, weight, samples):
    'calculating propagator'

    G = np.zeros([n])
    for i in range(0, samples):
        p = path(points, n)
        E = energy(p, potential)
        W = weight(E)
        position = p[0]
        idx = np.where(points == position)
        G[idx] += W
    return G

G = prop(x, pot, path_gen, E, W, N)


###repeating propogator for smaller samples

end = 2

Ns = np.logspace(start=1, stop= end, base=10, num= end)

Gs = np.zeros([len(Ns), n])
for j in range(0, len(Ns)):
    for i in Ns:
        Gs[j] = prop(x, pot, path_gen, E, W, int(i))



###normalisation function

def norm(array):
    'normalisation function'

    total = sum(array)
    normalised = array/total
    return normalised

Norm_G = norm(G)

Norm_Gs = np.zeros([len(Ns),n])
for i in range(0,len(Ns)):
    Norm_Gs[i] = norm(Gs[i])

y1 = Norm_G
ys = Norm_Gs


plt.plot(x, y1)
for i in range(0, len(Ns)):
    plt.plot(x, ys[i])
plt.legend()
plt.show()