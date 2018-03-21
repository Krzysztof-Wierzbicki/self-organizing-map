import numpy.random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse

from neuron import *
from rank import *

#stderr = sys.stderr
#log = open('out.log', 'w')
#sys.stderr = log

parser = argparse.ArgumentParser(description='Self organising map')
parser.add_argument('dataset', metavar='dataset', nargs='?', default='iris', help='dataset name, possible: iris')
parser.add_argument('-n', '--neurons', metavar='N', type=int, default=5, help='set number of neurons (centroids), default: 5')
parser.add_argument('-p', '--plot', action='store_true', help='set plotting on')
parser.add_argument('-c', '--count', type=int, default=100, help='set max number of epochs')

args = parser.parse_args()

if args.dataset == 'iris':
    filename = 'iris.data'

neuron_count = args.neurons
loop_max = args.count

data = pd.read_csv('iris.data', header=None).iloc[:, :2].values

mu = data[:, :].mean(axis=0)
sigma = data[:, :].std(axis=0)

# kohonen
def G(d, l):
    return np.e**(-(d**2)/(2.*l**2))
    
# here lambda is argument at which step occurs 
def H(d, l):
    return 1 if d<l else 0
    
NK = [NeuronKHN([rnd.normal(m, s) for m, s in zip(mu, sigma)]) for _ in range(neuron_count)]

NeuronKHN.set_epsilon(0.01)
NeuronKHN.set_lambda(0.0000001)
NeuronKHN.set_function(G)
#NeuronKHN.set_lambda(0.5)
#NeuronKHN.set_function(H)

fig = plt.figure(1)

errors = []
for epoch in range(40):
    rnd.shuffle(data)

    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i.pos[0] for i in NK], [i.pos[1] for i in NK], 'bo')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('Kohonen\nEpoch {}'.format(epoch))
    
    plt.pause(0.03)
    
    error = 0
    for x, y in data[:, :2]:
        rank = Rank()
        for n in NK:
            rank[n] = n.distance([x, y])
               
        bmu = rank.best() 
        w_0 = bmu.pos
        error += bmu.distance([x, y])
        for n in rank:
            n.update(w_0, [x, y])
            
    errors.append(error)
    
fig.clear()
plt.plot(errors, 'b-')
plt.xlabel('epoch')
plt.ylabel('quantization error')
plt.title('Error')
plt.show()

# neural gas
NG = [NeuronNG([rnd.normal(m, s) for m, s in zip(mu, sigma)]) for _ in range(neuron_count)]

NeuronNG.set_epsilon(0.01)
NeuronNG.set_lambda(0.5) 
NeuronNG.pre_count(neuron_count)

fig = plt.figure(2)

errors = []
for epoch in range(50):
    rnd.shuffle(data)

    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i.pos[0] for i in NG], [i.pos[1] for i in NG], 'ro')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('Neural gas\nEpoch {}'.format(epoch))
    
    plt.pause(0.03)
    
    error = 0
    for x, y in data[:, :2]:
        rank = Rank()
        for n in NG:
            rank[n] = n.distance([x, y])
            
        error += rank.best().distance([x, y])
        for i, n in enumerate(rank):
            n.update(i, [x, y])
            
    errors.append(error)
    
fig.clear()
plt.plot(errors, 'r-')
plt.xlabel('epoch')
plt.ylabel('quantization error')
plt.title('Error')
plt.show()

#k-means
C = [Centroid([rnd.normal(m, s) for m, s in zip(mu, sigma)]) for _ in range(neuron_count)]
      
fig = plt.figure(3)
      
errors = []
for epoch in range(loop_max):
    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i.pos[0] for i in C], [i.pos[1] for i in C], 'go')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('K-Means\nEpoch {}'.format(epoch))
    
    plt.pause(0.1)
    
    error = 0
    for x, y in data[:, :2]:
        rank = Rank()
        for c in C:
            rank[c] = c.distance([x, y])

        bmu = rank.best()
        bmu.add_datum([x, y])
        error += bmu.distance([x, y])
        
    for c in C:
        c.update()
        
    # exit condition
    if epoch > 0 and errors[-1] == error:
        break
        
    errors.append(error)

fig.clear()
plt.plot(errors, 'g-')
plt.xlabel('epoch')
plt.ylabel('quantization error')
plt.title('Error')
plt.show()

#sys.stderr = stderr      
#log.close()
