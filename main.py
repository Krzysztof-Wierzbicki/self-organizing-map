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

mu = data[:, :].mean(axis=0)
sigma = data[:, :].std(axis=0)

# kohonen
def G(d, l):
    return np.e**(-(d**2)/(2.*l**2))

def F(d, l):
    return 1/(1+np.e**(100*(d-l)))
    
def H(d, l):
    return 1 if d<l else 0
    
NK = [NeuronKHN([rnd.normal(m, s) for m, s in zip(mu, sigma)]) for _ in range(neuron_count)]

NeuronKHN.set_epsilon(0.01)
NeuronKHN.set_lambda(0.0000001)
NeuronKHN.set_function(G)
#NeuronKHN.set_lambda(0.5)
#NeuronKHN.set_function(H)

fig = plt.figure(1)

for epoch in range(40):
    rnd.shuffle(data)

    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i.pos[0] for i in NK], [i.pos[1] for i in NK], 'bo')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('Kohonen\nEpoch {}'.format(epoch))
    
    plt.pause(0.03)
    
    for x, y in data[:,:2]:
        rank = Rank()
        for n in NK:
            rank[n] = n.distance([x, y])
                
        w_0 = rank.best().pos
        for n in rank:
            n.update(w_0, [x, y])

# neural gas
NG = [NeuronNG([rnd.normal(m, s) for m, s in zip(mu, sigma)]) for _ in range(neuron_count)]

NeuronNG.set_epsilon(0.01)
NeuronNG.set_lambda(0.5) 
NeuronNG.pre_count(neuron_count)

fig = plt.figure(1)

for epoch in range(50):
    rnd.shuffle(data)

    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i.pos[0] for i in NG], [i.pos[1] for i in NG], 'ro')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('Neural gas\nEpoch {}'.format(epoch))
    
    plt.pause(0.03)
    
    for x, y in data[:,:2]:
        rank = Rank()
        for n in NG:
            rank[n] = n.distance([x, y])
        for i, n in enumerate(rank):
            n.update(i, [x, y])

#k-means
C = [Centroid([rnd.normal(m, s) for m, s in zip(mu, sigma)]) for _ in range(neuron_count)]
      
for epoch in range(30):
    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i.pos[0] for i in C], [i.pos[1] for i in C], 'go')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('K-Means\nEpoch {}'.format(epoch))
    
    plt.pause(0.1)
    
    for x, y in data[:,:2]:
        rank = Rank()
        for c in C:
            rank[c] = c.distance([x, y])

        rank.best().add_datum([x, y])
        
    for c in C:
        c.update()

#sys.stderr = stderr      
#log.close()
