import numpy.random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from neuron import *
from rank import *

#stderr = sys.stderr
#log = open('out.log', 'w')
#sys.stderr = log

data = pd.read_csv('iris.data', header=None).iloc[:, :2].values

neuron_count = 5

x_mu = data[:, 0].mean()
x_sigma = data[:, 0].std()
y_mu = data[:, 1].mean()
y_sigma = data[:, 1].std()

# kohonen
def G(d, l):
    return np.e**(-(d**2)/(2.*l**2))

def F(d, l):
    return 1/(1+np.e**(100*(d-l)))
    
def H(d, l):
    return 1 if d<l else 0
    
NK = [NeuronKHN(rnd.normal(x_mu, x_sigma), rnd.normal(y_mu, y_sigma)) for _ in range(neuron_count)]
    
for n in NK:
    n.set_rates(0.01, 0.3)
    n.set_function(F)

fig = plt.figure(1)

for epoch in range(100):
    rnd.shuffle(data)

    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i._x for i in NK], [i._y for i in NK], 'bo')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('Kohonen\nEpoch {}'.format(epoch))
    
    plt.pause(0.03)
    
    for x, y in data[:,:2]:
        rankK = {}
        for nk in NK:
            rankK[nk.distance(x, y)] = nk
        for i, k in enumerate(sorted(rankK.keys())):
            if i==0:
                w_0 = rankK[k].pos[:]
            rankK[k].update(w_0, x, y)

# neural gas
NG = [NeuronNG(rnd.normal(x_mu, x_sigma), rnd.normal(y_mu, y_sigma)) for _ in range(neuron_count)]

for n in NG:
    n.set_rates(0.01, 0.5) 
    n.pre_count(neuron_count)

fig = plt.figure(1)

for epoch in range(100):
    rnd.shuffle(data)

    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i._x for i in NG], [i._y for i in NG], 'ro')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('Neural gas\nEpoch {}'.format(epoch))
    
    plt.pause(0.03)
    
    for x, y in data[:,:2]:
        rankG = {}
        for ng in NG:
            rankG[ng.distance(x, y)] = ng
        for i, k in enumerate(sorted(rankG.keys())):
            rankG[k].update(i, x, y)

#k-means
C = [Centroid(rnd.normal(x_mu, x_sigma), rnd.normal(y_mu, y_sigma)) for _ in range(neuron_count)]
      
for epoch in range(100):
    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i._x for i in C], [i._y for i in C], 'go')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('K-Means\nEpoch {}'.format(epoch))
    
    plt.pause(0.1)
    
    for x, y in data[:,:2]:
        rank = {}
        for c in C:
            rank[c.distance(x, y)] = c

        lowest_distance = min(rank.keys())
        ld_count = list(rank.keys()).count(lowest_distance)
        random_key = sorted(rank.keys())[rnd.randint(ld_count)]
        rank[random_key].add_datum(x, y) #get value by key that is in position 0 in sorted list of keys
        
    for c in C:
        c.update()

#sys.stderr = stderr      
#log.close()
