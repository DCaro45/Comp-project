"""using metropolis to find probability density function for 2D potentials"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
from matplotlib import cm
import os

dir, file = os.path.split(__file__)

'''values'''
mass = 1   # setting mass to be 1

t_i = 0     # start time
t_f = 0.1     # finish time
div = 10  # division of time points (i.e. whole numbs, half, third etc.))

epsilon = 1  # change in delta_xs size from spatial lattice spacing
bins = 50      # number of bins for histogram

N_cor = 20        # number of paths to be skipped path set (due to correlation)
Therm = 5 * N_cor    # number of sweeps through path set
N_CF = 10 ** 4   # number of updates

u = 2            # potential parameter 1
l = 1            # potential parameter 2

'''determinants/shorthands'''
if t_f - t_i >= 1:
    n_tp = int( div * (t_f - t_i) + 1 )         # number of temporal points
    n_tl = int( div * (t_f - t_i) )             # number of temporal links
    a = (t_f - t_i) / n_tl                # size of time step
    t_points = np.linspace(t_i, t_f, n_tp)  # temporal lattice points
if t_f - t_i < 1:
    n_tp = div + 1          # number of temporal points
    n_tl = div              # number of temporal links
    a = (t_f - t_i) / div   # size of time step
    t_points = np.linspace(t_i, t_f, div + 1)  # temporal lattice points

m = mass           # shorthand for mass
nt = n_tp          # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
U = int(N_cor)    # shorthand for sweeps 2 (and integer data type)
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)
N = int(N_CF)      # shorthand for number of updates

print(
    'n = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) + ', ' +
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
    V = - 10 * x**2 + 8 * y**2 + 6 * x**4 - 3 * y**4 + 1/10 * x**6 + 1/10 * y**6
    return V


def pot4(x,y):
    """ an anharmonic potential with a variety of minima using sin and cos functions"""
    V = np.cos(x) + np.cos(y) + 1/5000000 * np.exp(x**2) + 1/5000000 * np.exp(y**2)
    return V


pot = pot1

x0 = -4
x1 = 4
X = np.linspace(x0, x1, 250)
Y = np.linspace(x0, x1, 250)
X, Y = np.meshgrid(X, Y)
Z = pot(X, Y)
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
    jp = (j-1) % nt
    jn = (j+1) % nt

    r = np.sqrt(x[j] ** 2 + y[j] ** 2)
    rp = np.sqrt(x[jp] ** 2 + y[jp] ** 2)
    rn = np.sqrt(x[jn] ** 2 + y[jn] ** 2)
    KE = m * r * (r - rp - rn) / (a ** 2)

    #KE_x = m * x[j] * (x[j] - x[jp] - x[jn]) / (a ** 2)
    #KE_y = m * y[j] * (y[j] - y[jp] - y[jn]) / (a ** 2)
    #KE = KE_x + KE_y

    PE = potential(x[j], y[j])
    E_tot = KE + PE
    Action = a * E_tot

    return Action


def Metropolis(path_x, path_y, potential):
    """creating the metropolis algorithm"""
    count = 0
    N = len(path_x)
    M = len(path_y)

    for i in range(nt):
        if N != nt or M != nt:
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

"""Initialising paths and trialing metropolis"""

px_1 = np.zeros([nt])
py_1 = np.zeros([nt])
px1, py1, count = Metropolis(px_1, py_1, pot)
print(px1, py1, count/nt, sep='\n')

"""Thermalising lattice"""

init_x = px_1
init_y = py_1
'''
for i in range(T):
    new_px, new_py, counts = Metropolis(init_x, init_y, pot)
    init_x = new_px
    init_y = new_py
'''


"""Generating paths and applying metropolis"""

all_px = np.zeros([N, nt])
all_py = np.zeros([N, nt])
all_px[0] = init_x
all_py[0] = init_y

t_counts = 0
for j in range(N - 1):
    start_px = all_px[j]
    start_py = all_py[j]
    for i in range(U):
        new_px, new_py, counts = Metropolis(start_px, start_py, pot)
        start_px = new_px
        start_py = new_py
        t_counts += counts
    all_px[j + 1] = start_px
    all_py[j + 1] = start_py
print('prop of changing point = ' + str(t_counts/(nt*U*N)))


"""All points from new_ps"""

xpos = np.zeros([N * nt])
ypos = np.zeros([N * nt])

k = 0
for i in range(N):
    for j in range(nt):
        xpos[k] = all_px[i][j]
        ypos[k] = all_py[i][j]
        k += 1
print('done')



"""Graphs"""

"Generating potential"
x0 = min(xpos)
x1 = max(xpos)
y0 = min(ypos)
y1 = max(ypos)

X = np.linspace(x0, x1, 100)
Y = np.linspace(y0, y1, 100)
X, Y = np.meshgrid(X, Y)
if pot == pot1:
    Z = pdf(X, Y)
    name = 'Harmonic'
elif pot == pot2:
    R = u/l
    name = 'Higgs'
else:
    Z = pot(X, Y)

"3D Hist"
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

if pot == pot1:
    hist, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins)
    Norm = np.max(Z)/np.max(hist) * hist
else:
    hist, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins, density=True)
    Norm = hist

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
fig.savefig(dir + '\\Images\\3Dhist_' + name + '-' + str(t_f) + 's.png')
plt.show()


"Contour Hist"
fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(title='Histogram Contour')

hist, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins)
hist = hist.T

x, y = np.meshgrid(xedges, yedges)
ax.pcolormesh(x, y, hist)

if pot == pot2:
    xs = np.linspace(- R, R, 100)
    plt.plot(xs, - (R**2 - xs**2) ** (1/2), 'k-')
    plt.plot(xs, (R**2 - xs**2) ** (1/2), 'k-')
fig.savefig(dir + '\\Images\\contour-hist_' + name + '-' + str(t_f) + 's.png')
plt.show()
