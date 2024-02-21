"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
from matplotlib import cm
import os


dir, file = os.path.split(__file__)

'''values'''
mass = 1   # setting mass to be 1

ti = 0     # start time
tf = 2     # finish time
div_t = 2  # division of time points (i.e. whole numbs, half, third etc.))

epsilon = 1.2  # change in delta_xs size from spatial lattice spacing
bins = 50     # number of bins for histogram

N_cor = 20        # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 4    # number of updates
Therm = 10 * N_cor    # number of sweeps through path set

u = 2            # potential parameter 1
l = 1            # potential parameter 2

'''determinants/shorthands'''
n_tp = div_t * (tf - ti) + 1          # number of temporal points
n_tl = div_t * (tf - ti)              # number of temporal links
a = (tf - ti) / n_tl                  # size of time step
t_points = np.linspace(ti, tf, int(n_tp))  # temporal lattice points

m = mass           # shorthand for mass
n = int(n_tp)          # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
U = int(N_cor)    # shorthand for sweeps 2 (and integer data type)
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)
N = int(N_CF)      # shorthand for number of updates

print(
    'nt = ' + str(n) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) + ', ' +
    'N_cor/Update = ' + str(U) + ', ' + 'S1 = ' + str(T) + ', ' + 'N_CF = ' + str(N)
      )


def pdf(x, y):
    """prob density function"""
    r = np.sqrt(x ** 2 + y ** 2)
    prob = np.exp(- r ** 2) / np.pi
    return prob


def pot1(x, y):
    """simple harmonic oscillator potential"""
    r = np.sqrt(x ** 2 + y ** 2)
    V = 1/2 * r ** 2
    return V


def pot2(x, y):
    """a simple potential analogue for the Higgs potential"""
    r = np.sqrt(x ** 2 + y ** 2)
    V = - 0.5 * u ** 2 * r ** 2 + 0.25 * l ** 2 * r ** 4
    return V


def pot3(x, y):
    """ an anharmonic potential with a variety of minima"""
    r = np.sqrt(x ** 2 + y ** 2)
    V = 1/2 * r ** 2 - 1/3 * r ** 2 + 1/4 * r ** 4 + 1/20 * r ** 5
    return V

V = pot1

x0 = -10
x1 = 10
X = np.linspace(x0, x1, 250)
Y = np.linspace(x0, x1, 250)
X, Y = np.meshgrid(X, Y)
Z = V(X, Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_formatter('{x:.02f}')
ax.tick_params(axis='z', labelcolor='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Potential')
plt.show()


def actn(x, y, j, potential):
    """calculating energies"""
    jp = (j-1) % n
    jn = (j+1) % n

    r = np.sqrt(x[j] ** 2 + y[j] ** 2)
    rp = np.sqrt(x[jp] ** 2 + y[jp] ** 2)
    rn = np.sqrt(x[jn] ** 2 + y[jn] ** 2)

    KE = m * r * (r - rp - rn) / (a ** 2)
    PE = potential(x[j], y[j])
    E_tot = KE + PE
    Action = a * E_tot

    return Action

def Metropolis(path_x, path_y, potential):
    """creating the metropolis algorithm"""
    count = 0
    N = len(path_x)
    M = len(path_y)

    for i in range(n):
        if N != n or M != n:
            print('error: path length not equal to n')
            break
        dx = rdm.uniform(-e, e)
        dy = rdm.uniform(-e, e)

        eval_px = path_x.copy()
        eval_py = path_y.copy()
        eval_px[i] = path_x[i] + dx
        eval_py[i] = path_y[i] + dy

        S1 = actn(path_x, path_y, i, potential)
        S2 = actn(eval_px, eval_py, i,  potential)
        dS = S2 - S1

        r = rdm.random()
        W = np.exp(-dS)
        if dS < 0 or W > r:
            path_x = eval_px
            path_y = eval_py
            count += 1

    return path_x, path_y, count

pot = pot1

px_1 = np.zeros([n])
py_1 = np.zeros([n])
px1, py1, count = Metropolis(px_1, py_1, pot)
print(px1, py1, count/n, sep='\n')

init_x = px_1
init_y = py_1

'''
"""Thermalising lattice"""
for i in range(T):
    new_px, new_py, counts = Metropolis(init_x, init_y, pot)
    init_x = new_px
    init_y = new_py
'''

"""generating paths and applying metropolis"""
all_ps_x = np.zeros([N, n])
all_ps_y = np.zeros([N, n])
t_counts = 0
for j in range(N):
    if j == 0:
        start_px = init_x
        start_py = init_y
    else:
        start_px = all_ps_x[j - 1]
        start_py = all_ps_y[j - 1]
    for i in range(U):
        new_px, new_py, counts = Metropolis(start_px, start_py, pot)
        start_px = new_px
        start_py = new_py
        t_counts += counts
    all_ps_x[j] = start_px
    all_ps_y[j] = start_py
print('prop of changing point = ' + str(t_counts/(n*U*N)))


"""all points from new_ps"""

xpos = np.zeros([N * n])
ypos = np.zeros([N * n])
k = 0
for i in range(N):
    for j in range(n):
        xpos[k] = all_ps_x[i][j]
        ypos[k] = all_ps_y[i][j]
        k += 1
print('done')


x0 = min(xpos)
x1 = max(xpos)
y0 = min(ypos)
y1 = max(ypos)

X = np.linspace(x0, x1, 100)
Y = np.linspace(y0, y1, 100)
X, Y = np.meshgrid(X, Y)
Z = pdf(X, Y)

R = u/l

"3D Hist"
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

hist, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins) #, density=True)
Norm = np.max(Z)/np.max(hist) * hist

x, y = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
x = x.flatten()/2
y = y.flatten()/2
z = np.zeros_like(x)

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = Norm.flatten()

cmap = cm.get_cmap('viridis')
max_height = np.max(dz)
min_height = np.min(dz)
rgba = [cmap((k-min_height)/max_height) for k in dz]

ax.bar3d(x, y, z, dx, dy, dz, color=rgba, zsort='average')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.3, linewidth=0, antialiased=False)
ax.zaxis.set_major_formatter('{x:.02f}')
ax.tick_params(axis='z', labelcolor='red')

plt.title("The Probability Density for a Harmonic Oscillator Potential")
plt.xlabel("x position")
plt.ylabel("y position")
ax.set_zlabel("Probability Density")
#txt=("A 2D Histogram of the location of points in paths produced via the Metropolis Algorithm within a Harmonic "
#     "Oscillator Potential."
#     "The 3D surface plot is the analytic solution to probability density function of the potential."
#     )
#plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
#fig.savefig(dir + '\\Images\\3Dhist.png')
plt.show()


"Contour Hist"
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(title='Histogram Contour')

hist, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins)
hist = hist.T

x, y = np.meshgrid(xedges, yedges)
ax.pcolormesh(x, y, hist)

xs = np.linspace(- R, R, 100)
if pot == pot2:
    plt.plot(xs, - (R**2 - xs**2) ** (1/2), 'k-')
    plt.plot(xs, (R**2 - xs**2) ** (1/2), 'k-')
#fig.savefig(dir + '\\Images\\contour-hist_Higgs.png')
plt.show()
