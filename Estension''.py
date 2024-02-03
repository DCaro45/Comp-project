"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os
from matplotlib.image import NonUniformImage


'''values'''
mass = 1   # setting mass to be 1

ti = 0     # start time
tf = 4     # finish time
div_t = 2  # division of time points (i.e whole numbs, half, third etc))

epsilon = 1.5  # change in delta_xs size from spatial lattice spacing
bins = 100     # number of bins for histogram

N_cor = 25        # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 3    # number of updates

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

def pot1(r):
    """simple harmonic oscillator potential"""
    V = 1/2 * r ** 2
    return V

def pot2(r):
    """a simple potential analogue for the Higgs potential"""
    u = 3.5
    l = 1
    exp = 2
    V = - 0.5 * u ** 2 * r ** exp + 0.25 * l ** 2 * r ** (2 * exp)
    return V

def pdf(r):
    """prob density function"""
    prob = ( np.exp( - r ** 2 / 2 ) / ( np.pi ** (1/4) ) ) ** 2
    return prob

V = pot1

theta_2 = np.linspace(0, 2 * np.pi, 10**2)
r = np.linspace(0, 3.5, 10**2)

X = np.outer(r, np.cos(theta_2))
Y = np.outer(r, np.sin(theta_2))

X, Y = np.meshgrid(X, Y)
#R = V(np.sqrt(X**2 + Y**2))
Z = [r,r,r]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_formatter('{x:.02f}')
ax.tick_params(axis='z', labelcolor='red')
#plt.show()

pot = pot1

def actn(R, j, potential):
    """calculating energies"""
    jp = (j-1) % nt
    jn = (j+1) % nt

    r = R[j]
    rp = R[jp]
    rn = R[jn]

    KE = m * r * (r - rp - rn) / (a ** 2)
    PE = potential(r)
    E_tot = KE + PE
    Action = a * E_tot

    return Action

def Metropolis(path_r, potential):
    """creating the metropolis algorithm"""
    count = 0

    for j, R in enumerate(path_r):
        dR = rdm.uniform(-e, e)

        eval_pr = path_r.copy()
        eval_pr[j] = R + dR

        S1 = actn(path_r, j, potential)
        S2 = actn(eval_pr,j, potential)
        ds = S2 - S1

        r = rdm.random()
        W = np.exp(-ds)
        if ds < 0 or W > r:
            path_r = eval_pr
            count += 1

    return path_r, count

pr_1 = [0 for r in range(nt)]
pr1, count = Metropolis(pr_1, pot)
print(pr1, count/(nt))

"""generating paths and applying metropolis"""
all_ps_r = []
t_counts = 0
for j in range(N_CF):
    start_r = pr_1
    for i in range(U):
        new_r, counts = Metropolis(start_r, pot)
        start_r = new_r
        t_counts += counts
    all_ps_r.append(start_r)

print('prop of changing point = ' + str(t_counts/((nt**2)*U*N_CF)))

"""all points fromn new_ps"""
ln = len(all_ps_r)
rpos= np.zeros([ln * nt])
k = 0
for i in range(ln):
    for j in range(nt):
        rpos[k] = all_ps_r[i][j]
        k += 1

print('done')
L = len(rpos)
r = rpos.flatten()
L_2 = len(r)
print(L)

theta = np.linspace(0, 2 * np.pi, L_2)
r = np.linspace(0, 3.5, L_2)

xs = np.outer(rpos, np.cos(theta))
ys = np.outer(rpos, np.sin(theta))
xs = X.flatten()
ys = Y.flatten()
print(X)

#xpos, ypos = np.meshgrid(X, Y)
#print(xpos)


r0 = min(rpos)
r1 = max(rpos)

X = np.linspace(r0, r1, 100)
Y = np.linspace(r0, r1, 100)
X, Y = np.meshgrid(X, Y)
#Z = pdf(rpos)
R = np.sqrt(X**2 + Y**2)
Z = pot(R)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(xs, ys, bins=bins)#, density=True)
Norm = np.max(Z)/np.max(hist) * hist
Norm = hist

x, y = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

x = x.flatten()/2
y = y.flatten()/2
z = np.zeros_like (x)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = Norm.flatten()

cmap = cm.get_cmap('viridis')
max_height = np.max(dz)
min_height = np.min(dz)
rgba = [cmap((k-min_height)/max_height) for k in dz]

ax.bar3d(x, y, z, dx, dy, dz, color=rgba, zsort='average')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.3, linewidth=0, antialiased=False)
#ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
ax.tick_params(axis='z', labelcolor='red')

#plt.title("X vs. Y Amplitudes for ____ Data")
plt.xlabel("My X data source")
plt.ylabel("My Y data source")
dir, file = os.path.split(__file__)
#fig.savefig(dir + '\\Images\\3Dhist.png')
plt.show()


x = xpos
y = ypos

r = 3.5
xs = np.linspace(- r, r, 100)


H, xedges, yedges = np.histogram2d(x, y, bins=bins)
H = H.T

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='imshow: square bins')
plt.imshow(H, interpolation='nearest', origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax = fig.add_subplot(132, title='pcolormesh: actual edges',
        aspect='equal')
X, Y = np.meshgrid(xedges, yedges)
ax.pcolormesh(X, Y, H)
plt.plot(xs, - (r**2 - xs**2) ** (1/2), 'k-')
plt.plot(xs, (r**2 - xs**2) ** (1/2), 'k-')

ax = fig.add_subplot(133, title='NonUniformImage: interpolated',
        aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear')
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
im.set_data(xcenters, ycenters, H)
ax.add_image(im)
plt.show()
