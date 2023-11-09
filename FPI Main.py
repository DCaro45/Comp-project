import numpy as np
import matplotlib.pyplot as plt
import random as rdm

'''values'''
mass = 1

x0 = -2       #lower bound
x1 = 2        #upper bound
ti = 0        #start time
tf = 5        #finish time

n_t = 7                         #Number of time points
divs_x = 4                      #Division of space points (i.e whole numbs, half, third etc)
n_x = divs_x * (x1 - x0) + 1    #number of spatial points


N= int(10e+8)                   #number of samples for prop calc
end = 5                         #10^(end) samples
nbr = 2                         #number of graphs

'''determinants/shorthands'''

m = mass                       #shorthand for mass

a = (tf-ti)/(n_t-1)                  #size of time step
t_points = np.linspace(ti, tf, n_t)  #time lattice points
x_points = np.linspace(x0, x1, n_x)  #spacial lattice points
'check that the t_step in t_points = a'

n = n_x                          #shorthand for no.x points
x = x_points                     #shorthand for lattice points


'''path function'''
def path_gen(xs, n):
    'path generator'

    path = np.zeros([n])
    path[0]=path[n-1]=rdm.choice(xs)
    for i in range(1,n-1):
        path[i]=rdm.choice(xs)
    return path
p_1 = path_gen(x, n)


'''potential energy functions'''
def pot(x):
    'simple harmonic oscillator potential'
    V = 1/2*(x)**2
    return V


'''energy function'''
def Engy(path, potential):
    'calculating energies'
    E_path = [0]
    for i in range(0, n-1):
        KE = m/2*((path[i+1]-path[i])/a)**2
        PE = potential((path[i+1]+path[i])/2)
        E_tot = KE + PE
        E_path+=  E_tot
    return E_path

'''weighting function'''
def Wght(energy):
    'calculating weight'
    weight = np.exp(-a*energy)
    return weight


'''propogator function'''
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

'Calculating propogator value'
G = prop(x, pot, path_gen, Engy, Wght, N)


'''repeating propagator for smaller samples'''
Ns = np.logspace(start=0, stop= end, base=2, num=nbr)
lng = len(Ns)


Gs = np.zeros([lng, n])
for j in range(0, lng):
    for i in Ns:
        Gs[j] = prop(x, pot, path_gen, Engy, Wght, int(i))


'''normalisation function'''
def norm(array):
    'normalisation function'
    total = sum(array)
    normalised = array/total
    return normalised


'''plotting graphs'''
'Normalising Gs'
Norm_G = norm(G)

Norm_Gs = np.zeros([lng,n])
for i in range(0,lng):
    Norm_Gs[i] = norm(Gs[i])

y1 = Norm_G
ys = Norm_Gs

plt.figure()
plt.plot(x, y1)
plt.show()

As = np.linspace(int(1/lng), 1, lng)

plt.figure()
for j in range(0, lng):
    plt.plot(x, ys[j], label=Ns[j], alpha = As[j])
plt.legend()
plt.show()



'''ground state w_fn analystic equation'''
def pdf(x):
    'prob density function'

    prob = ( np.exp(-(x**2/2)) / np.pi**(1/4) ) ** 2
    return prob


'''plot of FPI and standard formulation'''

'calculate potential and analytic pdf'
pdf_A = pdf(x)
y2 = norm(pdf_A)

l = 100 * (x1 - x0) + 1
xs = np.linspace(-2, 2, l)
ys = pot(xs)

'plotting graphs'
plt.figure()
plt.plot(x , y1, label = 'FPI', color = 'k')
plt.plot(x , y2, label = 'PDF', color = 'tab:orange' )
plt.plot(xs, ys, label = 'Potential', color = 'tab:blue')
plt.legend()
plt.grid()
plt.xlim(-2, 2)
plt.xlabel('position')
plt.ylabel('probability')
plt.ylim(0, max(y1) + 0.1*max(y1))
plt.show()





