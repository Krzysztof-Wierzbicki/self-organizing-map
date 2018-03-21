import numpy.random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuron import *
from rank import *
from parsing import *

#stderr = sys.stderr
#log = open('out.log', 'w')
#sys.stderr = log

def normalize(data):
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    for i in range(data.shape[1]):
        data[:, i] -= mean[i]
        data[:, i] /= std[i]

def make_neurons(cls, neuron_count, data):
    '''create neurons using normal distribution for given data'''
    return [cls([rnd.normal(m, s) for m, s in zip(data[:, :-1].mean(axis=0), data[:, :-1].std(axis=0))]) for _ in range(neuron_count)]
    
def plot(data, data_name, neurons, algorithm, epoch, color):
    '''plot data and central structures'''
    fig.clear()
      
    plt.plot(data[:, 0], data[:, 1], 'k.')
    plt.plot([i.pos[0] for i in neurons], [i.pos[1] for i in neurons], '{}o'.format(color))
    
    if data_name == 'iris':
        plt.xlabel('sepal width')
        plt.ylabel('sepal length')
        
    plt.title('{}\nepoch {}'.format(algorithm, epoch))
    
    plt.pause(0.03)
    
    
def plot_error(errors, color):
    '''plot error against time'''
    fig.clear()
    plt.plot(errors, '{}-'.format(color))
    plt.xlabel('epoch')
    plt.ylabel('quantization error')
    plt.title('Error')
    plt.show()
    
def kohonen_fit(data, neurons):
    error = 0
    for i, xy in enumerate(data[:, :-1]):
        rank = Rank()
        for n in neurons:
            rank[n] = n.distance(xy)
               
        bmu = rank.best()
        data[i, -1] = bmu.id
        w_0 = bmu.pos
        error += bmu.distance(xy)
        for n in rank:
            n.update(w_0, xy)
            
    return error

def neural_gas_fit(data, neurons):
    error = 0
    for i, xy in enumerate(data[:, :-1]):
        rank = Rank()
        for n in neurons:
            rank[n] = n.distance(xy)
            
        bmu = rank.best()
        data[i, -1] = bmu.id
        error += bmu.distance(xy)
        for j, n in enumerate(rank):
            n.update(j, xy)
            
    return error

def k_means_fit(data, centroids):
    error = 0
    for i, xy in enumerate(data[:, :-1]):
        rank = Rank()
        for c in centroids:
            rank[c] = c.distance(xy)

        bmu = rank.best()
        data[i, -1] = bmu.id
        bmu.add_datum(xy)
        error += bmu.distance(xy)
        
    for c in centroids:
        c.update()
        
    return error

# kohonen functions
def g(d, l):
    return np.e**(-(d**2)/(2.*l**2))
    
# here lambda is argument at which step occurs 
def h(d, l):
    return 1 if d<l else 0

# dictionaries for setting constants
algorithms = {'kohonen':kohonen_fit, 'neural-gas':neural_gas_fit, 'k-means':k_means_fit}
datasets = {'iris':'iris.data'}
central_structures = {'kohonen':NeuronKHN, 'neural-gas':NeuronNG, 'k-means':Centroid}
funcs = {'step':h, 'other':g}
columns_d = {'iris':[0, 1]}

# create argument parser, parse arguments and set constants
P = Parsing()
P.init_parser(algorithms.keys(), datasets.keys(), funcs.keys())
P.create_parser()
P.parse()

no_errors    = P.get_no_errors()
no_plot      = P.get_no_plot()
if no_plot:
    columns = columns_d[P.get_dataset()]
else:
    columns = slice(0, -1)

lamb    = P.get_lambda()
epsilon = P.get_epsilon()

dataset  = P.get_dataset()
filename = datasets[dataset]

neuron_count = P.get_neurons()
loop_max     = P.get_count()

algorithm  = P.get_algorithm()
unit_class = central_structures[algorithm]
fit        = algorithms[algorithm]
G          = funcs[P.get_function()]

# read data, add ownership column, normalize
data   = pd.read_csv(filename, header=None).iloc[:, columns].values
normalize(data)
filler = np.zeros([data.shape[0], 1], dtype=int)
data   = np.append(data, filler, axis=1)

# make and initialize neurons/centroids
N = make_neurons(unit_class, neuron_count, data)

if algorithm == 'kohonen':
    NeuronKHN.set_epsilon(epsilon)
    NeuronKHN.set_lambda(lamb)
    NeuronKHN.set_function(G)
elif algorithm == 'neural-gas':
    NeuronNG.set_epsilon(epsilon)
    NeuronNG.set_lambda(lamb) 
    NeuronNG.pre_count(neuron_count)

# main
fig = plt.figure()

errors = []
for epoch in range(loop_max):
    rnd.shuffle(data)

    if no_plot:
        plot(data, dataset, N, algorithm, epoch, 'b')
    
    prev_ownership = np.array(data[:, -1], copy=True)
    error = fit(data, N)      
    errors.append(error)
    
#    if np.array_equal(prev_ownership, data[:, -1]):
#        break
    
if no_errors:
    plot_error(np.sqrt(np.array(errors)**2/data.shape[0]), 'b')

#sys.stderr = stderr      
#log.close()
