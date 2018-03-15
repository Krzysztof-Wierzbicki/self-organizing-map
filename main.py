import numpy.random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from neuron import *

#stderr = sys.stderr
#log = open('out.log', 'w')
#sys.stderr = log

data = pd.read_csv('iris.data', header=None).iloc[:, :2].values

neuron_count = 5

x_mu = data[:, 0].mean()
x_sigma = data[:, 0].std()
y_mu = data[:, 1].mean()
y_sigma = data[:, 1].std()

#neural gas and kohonen

NG = [NeuronNG(rnd.normal(x_mu, x_sigma), rnd.normal(y_mu, y_sigma)) for _ in range(neuron_count)]
NK = [NeuronKHN(rnd.normal(x_mu, x_sigma), rnd.normal(y_mu, y_sigma)) for _ in range(neuron_count)]

for n in NG:
    n.set_rates(0.01, 0.5) 
    n.pre_count(neuron_count)
    
def G(d, l):
    return np.e**(-(d**2)/(2.*l**2))

def F(d, l):
    return 1/(1+np.e**(100*(d-l)))
    
def H(d, l):
    return 1 if d<l else 0
    
for n in NK:
    n.set_rates(0.01, 0.3)
    n.set_function(F)

fig = plt.figure(1)

for epoch in range(100):
    rnd.shuffle(data)

    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i._x for i in NG], [i._y for i in NG], 'ro')
    plt.plot([i._x for i in NK], [i._y for i in NK], 'bo')
    
    plt.xlabel('sepal width')
    plt.ylabel('sepal length')
    plt.title('Epoch {}'.format(epoch))
    
    plt.pause(0.03)
    
    for x, y in data[:,:2]:
        rankG = {}  #neural gas
        rankK = {}  #kohonen
        for ng, nk in zip(NG, NK):
            rankG[ng.distance(x, y)] = ng
            rankK[nk.distance(x, y)] = nk
        for i, k in enumerate(sorted(rankG.keys())):
            rankG[k].update(i, x, y)
        for i, k in enumerate(sorted(rankK.keys())):
            if i==0:
                w_0 = rankK[k].pos[:]
            rankK[k].update(w_0, x, y)

#sys.stderr = stderr      
#log.close()
