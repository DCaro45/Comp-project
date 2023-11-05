import numpy as np
import matplotlib.pyplot as plt
import random as rdm

##values
number = 2  #Division of ticks (i.e whole numbs, half, third etc)
x1 = 2      #upper bound
x0 = -x1    #lower bound
ti = 0      #start time
tf = 5      #finish time


##determinants

n = number * (x1 - x0) + 1     #number of steps
n = n
a = (tf-ti)/(n-1)                  #time step
points = np.linspace(x0,x1,n)  #spacial lattice points

x = points       #shorthand for lattice points



##path

def path_gen(xs, n, x0):
    path = np.zeros([n])
    path[0]=path[n-1] = x0
    for i in range(1,n-1):
        path[i]=rdm.choice(xs)
    return path

p_1 = path_gen(x, n, x[2])


###graph

samples = 5

#time points
ts = np.linspace(ti, tf, n)

plt.figure(figsize = [3, 4])

for i in range(0,samples):
    p = path_gen(x, n, x[4])
    plt.plot(p, ts)

plt.grid(axis = 'x')
plt.show()