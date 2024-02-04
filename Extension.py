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
N_CF = 10 ** 4      # number of updates

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

def pot1(x, y):
    """simple harmonic oscillator potential"""
    r = np.sqrt(x ** 2 + y ** 2)
    V = 1/2 * r ** 2
    return V

def pot2(x, y):
    """a simple potential analogue for the Higgs potential"""
    u = 3.5
    l = 1
    r = np.sqrt(x ** 2 + y ** 2)
    V = - 0.5 * u ** 2 * r ** 2 + 0.25 * l ** 2 * r ** 4
    return V

def pdf(x, y):
    """prob density function"""
    r = np.sqrt(x ** 2 + y ** 2)
    #prob = np.exp(- r ** 2) /np.pi
    prob = np.exp( - r ** 2)/(np.pi ** (1/2))
    return prob

V = pot1

x0 = -4
x1 = 4

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
#plt.show()

pot = pot1

def actn(x, y, j, k, potential):
    """calculating energies"""
    jp = (j-1) % nt
    jn = (j+1) % nt
    kp = (k-1) % nt
    kn = (k+1) % nt

    r = np.sqrt(x[j] ** 2 + y[k] ** 2)
    rp = np.sqrt(x[jp] ** 2 + y[kp] ** 2)
    rn = np.sqrt(x[jn] ** 2 + y[kn] ** 2)

    KE = m * r * (r - rp - rn) / (a ** 2)
    PE = potential(x[j], y[k])
    E_tot = KE + PE
    Action = a * E_tot

    return Action

def Metropolis(path_x, path_y, potential):
    """creating the metropolis algorithm"""
    count = 0

    for j, x in enumerate(path_x):
        for k, y in enumerate(path_y):
            dx = rdm.uniform(-e, e)
            dy = rdm.uniform(-e, e)

            eval_px = path_x.copy()
            eval_py = path_y.copy()

            eval_px[j] = x + dx
            eval_py[k] = y + dy

            S1 = actn(path_x, path_y, j, k, potential)
            S2 = actn(eval_px, eval_py, j, k,  potential)
            ds = S2 - S1

            r = rdm.random()
            W = np.exp(-ds)
            if ds < 0 or W > r:
                path_x = eval_px
                path_y = eval_py
                count += 1

    return path_x, path_y, count

px_1 = [0 for x in range(nt)]
py_1 = [0 for y in range(nt)]
px1, py1, count = Metropolis(px_1, py_1, pot)
print(px1, py1, count/(nt**2))

"""Thermalising lattice"""
init_x = px_1
init_y = py_1
array_x = [init_x]
array_y = [init_y]
for i in range(T):
    new_px, new_py, counts = Metropolis(array_x[-1], array_y[-1], pot)
    array_x.append(new_px)
    array_y.append(new_py)

"""generating paths and applying metropolis"""
all_ps_x = []
all_ps_y = []
t_counts = 0
for j in range(N_CF):
    start_px = init_x
    start_py = init_y
    for i in range(U):
        new_px, new_py, counts = Metropolis(start_px, start_py, pot)
        start_px = new_px
        start_py = new_py
        t_counts += counts
    all_ps_x.append(start_px)
    all_ps_y.append(start_py)

print('prop of changing point = ' + str(t_counts/((nt**2)*U*N_CF)))

"""all points fromn new_ps"""
ln = len(all_ps_x)
xpos= np.zeros([ln * nt])
ypos = np.zeros([ln * nt])
k = 0
for i in range(ln):
    for j in range(nt):
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
#Z = pot(X, Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(xpos, ypos, bins=(bins,bins))#, density=True)
Norm = np.max(Z)/np.max(hist) * hist

#print(x0, min(xedges), x1,  max(xedges))
#print(y0, min(yedges), y1, max(yedges))

x, y = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

x = x.flatten()/2
y = y.flatten()/2
z = np.zeros_like (x)

#print(x0, min(x), x1, max(x))
#print(y0, min(y), y1, max(y))

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

plt.title("The Probability Density for a Harmonic Oscillator Potential")
plt.xlabel("x position")
plt.ylabel("y position")
ax.set_zlabel("Probability Density")
dir, file = os.path.split(__file__)
#fig.savefig(dir + '\\Images\\3Dhist.png')
#txt=("A 2D Histogram of the location of points in paths produced via the Metropolis Algorithm within a Harmonic "
#     "Oscillator Potential."
#     "The 3D surface plot is the analytic solution to probability density function of the potential."
#     )
#plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()


x = xpos
y = ypos
r = 3.5

xs = np.linspace(- r, r, 100)

H, xedges, yedges = np.histogram2d(x, y, bins=bins)
H = H.T

fig = plt.figure(figsize=(6, 6))
#ax = fig.add_subplot(131, title='imshow: square bins')
#plt.imshow(H, interpolation='nearest', origin='lower',
 #       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax = fig.add_subplot(title='Histogram Contour')
#        ,aspect='equal')
X, Y = np.meshgrid(xedges, yedges)
ax.pcolormesh(X, Y, H)
#plt.plot(xs, - (r**2 - xs**2) ** (1/2), 'k-')
#plt.plot(xs, (r**2 - xs**2) ** (1/2), 'k-')

#ax = fig.add_subplot(133, title='NonUniformImage: interpolated',
#        aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear')
#xcenters = (xedges[:-1] + xedges[1:]) / 2
#ycenters = (yedges[:-1] + yedges[1:]) / 2
#im.set_data(xcenters, ycenters, H)
#ax.add_image(im)
plt.show()

