import numpy as np

class Unit:

    def __init__(self, coords):
        self._coords = coords
        
    def distance(self, point):
        return np.sqrt(sum([(a-b)**2 for a, b in zip(self._coords, point)]))
        
    def __repr__(self):
        string = "(" + ", ".join(["{}" for _ in self._coords]) + ")"
        return string.format(*self._coords)
        
    @property
    def id(self):
        return id(self)
        
    @property
    def pos(self):
        return self._coords
        
    @pos.setter
    def pos(self, value):
        self._coords = value
        
class Centroid(Unit):

    def __init__(self, coords):
        self._data_sum = [0 for _ in coords]
        self._data_count = 0
        super().__init__(coords)
        
    def add_datum(self, point):
        for i, x_i in enumerate(point):
            self._data_sum[i] += x_i
        self._data_count += 1
        
    def update(self):
        if self._data_count > 0:
            self._coords = [d/self._data_count for d in self._data_sum]
            self._data_sum = [0 for _ in self._data_sum]
            self._data_count = 0

class Neuron(Unit):

    @classmethod        
    def set_lambda(cls, lamb):
        cls._lambda = lamb
        
    @classmethod
    def set_epsilon(cls, epsilon):
        cls._epsilon = epsilon
        
class NeuronNG(Neuron):
      
    _coefficient = []
      
    @staticmethod
    def pre_count(n):
        NeuronNG._coefficient = [NeuronNG._epsilon * np.e**(-i/NeuronNG._lambda) for i in range(n)]
    
    def update(self, k, point):
        self._coords = [c + (self._coefficient[k] * (point[i] - c)) for i, c in enumerate(self._coords)]
        
class NeuronKHN(Neuron):

    @staticmethod
    def set_function(g):
        NeuronKHN._g = g
        
    def update(self, w_0, point):
        EGd = self._epsilon * NeuronKHN._g(self.distance(w_0), l=self._lambda)
        self._coords = [c + (EGd * (point[i] - c)) for i, c in enumerate(self._coords)]

