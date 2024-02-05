"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

'''values'''
mass = 1   # setting mass to be 1

ti = 0     # start time
tf = 20     # finish time
div_t = 2   # division of time points (i.e whole numbs, half, third etc))

epsilon = 1.4  # change in delta_xs size from spatial lattice spacing
N_cor = 25        # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 5       # number of updates

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
        jn = (j+n) % nt
        g += x[j] * x[jn]
    return g/nt

def compute_G2(x,n):
    g = 0
    for j in range(nt):
        jn = (j+n) % nt
        g += x[j] ** 3 * x[jn] ** 3
    return g/nt

def avg(p):
    Av = sum(p)/len(p)
    return Av

def sdev(p):
    P = np.array(p)
    sd = np.absolute(avg(P) ** 2 - avg(P ** 2)) ** (1/2)
    return sd

def delta_E(prop):
    G = np.array(prop)
    dE = np.log(np.absolute(G[:-1]/G[1:])) / a
    return dE

'''
def delta_E2(prop):
    avgG = avg(G)
    dE = np.log(np.absolute(avgG[:-1]/avgG[1:])) / a
    return dE
'''

def bootstrap(G):
    G_bootstrap = []
    L = len(G)
    for i in range(L):
        alpha = rdm.randint(0, L-1)
        G_bootstrap.append(G[alpha])
    return G_bootstrap

def bin(G, number):
    G_binned = []
    binsize = int(N_CF/number)
    for i in range(0, N_CF, binsize):
        G_avg = 0
        for j in range(binsize):
            if i+j >= N_CF:
                break
            G_avg += G[i+j]
        G_avg = G_avg/binsize
        G_binned.append(G_avg)
    return G_binned

compute_G = compute_G2

p_1 = [0 for t in range(nt)]
p_2 = [np.random.uniform(-4, 4) for t in range(nt)]
p1, count = Metropolis(p_1, pot)

G = compute_G(p_2, 0)
#print(G)

"""Thermalising lattice"""
init = p_1
for i in range(T):
    x, count = Metropolis(init, pot)
    init = x
therm = init

"""generating array of G values"""
G = np.zeros([N_CF, nt])
count = 0
x = therm
for alpha in range(N_CF):
    for j in range(U):
        new_x, c = Metropolis(x, pot)
        x = new_x
        count += c
    for n in range(nt):
        G[alpha][n] = compute_G(x,n)
print('prop of changing point = ' + str(count/(nt*U*N_CF)))
print('done G')

"""averaging G values"""
Av_G = np.zeros([nt])
for n in range(nt):
    avg_G = 0
    for alpha in range(N_CF):
        avg_G += G[alpha][n]
    avg_G = avg_G/N_CF
    Av_G[n] = avg_G
Avg_G = avg(G)

'Binning G values'
binned_G = bin(G, 20)
B = binned_G
Avg_B = avg(B)



"""Calculating delta_E"""
dE = delta_E(Av_G)       # delta_E for average G
dE_1 = delta_E(G[0])     # delta_E for first G
dE_2 = delta_E(Avg_B)    # delta_E for binned G
dE_3 = delta_E(Avg_G)    # delta_E for average G (using function)


print('done dE')

"""Calculating errors"""
'Bootstrap'
n = 10e+0
dE_bootstrap = np.zeros([int(n), nt - 1])
for i in range(int(n)):
    G_bootstrap = bootstrap(G)
    Avg_G = avg(G_bootstrap)
    dE_bootstrap[i] = delta_E(Avg_G)
dE_avg = avg(dE_bootstrap)
dE_sd = sdev(dE_bootstrap)

#print('G(%d) = %g' % (n, avg_G) + ', ' + str(count/(nt*U*N_CF)))
#print(Av_G)

#print('avg G\n', avg(G))
#print('Delta E\n', delta_E(G))
#print('avg G = ' + str(Av_G) + str(sdev(Av_G)))

'Binned'
nums = [1.5**i for i in range(1, 18)]
print(nums)
sd_bin = np.zeros([len(nums), nt - 1])
for i, n in enumerate(nums):
    b = bin(G, n)
    '''
    avg_b = avg(b)
    deltaE = delta_E2(avg_b)
    sd = sdev(deltaE)
    sd_bin[i] = sdev(deltaE)
    '''
    'Bootstrap'
    n = 10e+1
    dE_bootstrap = np.zeros([int(n), nt - 1])
    for j in range(int(n)):
        G_bootstrap = bootstrap(b)
        Avg_G = avg(G_bootstrap)
        dE_bootstrap[j] = delta_E(Avg_G)
    sd = sdev(dE_bootstrap)
    sd_bin[i] = sd
#plt.plot(nums, sd_bin)
#plt.show()

print('done boot')



"""Plotting"""

dE_analytic = [1 for t in range(nt-1)]
ts = t[:-1]
#print(len(dE_2), len(ts))


plt.figure(figsize=[8, 4])
plt.plot(ts, dE_analytic, linestyle='--', color='black')
#plt.scatter(ts, dE, color='red', alpha=0.5, label='Delta_E')
#plt.scatter(ts, dE_1, color='blue', alpha=0.2, label='Delta_E calculated from first G values')
#plt.scatter(ts, dE_2, color='orange', label='Delta_E using binned G')
#plt.scatter(ts, dE_3, color='orange', label='Delta_E')
#plt.errorbar(ts, dE_avg, yerr=dE_sd, color='green', fmt='o', capsize=4, elinewidth=1, label='Delta_E using bootstrap method')
plt.errorbar(ts, dE_avg, yerr=dE_sd, color='blue', fmt='o', capsize=4, elinewidth=1, label='Delta_E from bootstrap method using x^3')


plt.xlabel('t')
plt.ylabel('$\Delta$E(t)')
plt.legend()
plt.title('Numerically Evaluated Energy Difference Between the Ground and First Excited State')
#txt = ('The energy difference as a function of time calculated from the value of the propagator of paths produced by '
#       'the Metropolis Algorithm within a Harmonic Oscillator Potential'
#       )
#plt.figtext(0.5, 0.2, txt, wrap=True, horizontalalignment='center', fontsize=12)

dir, file = os.path.split(__file__)
plt.savefig(dir + '\\Images\\delta_E_G2.png')
plt.show()



