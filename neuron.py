import numpy as np

class Unit:

    def __init__(self, x, y):
        self._x = x
        self._y = y
        
    def distance(self, x, y):
        return np.sqrt((self._x-x)**2 + (self._y-y)**2)
        
    def __repr__(self):
        return "({}, {})".format(self._x, self._y)
        
    @property
    def id(self):
        return id(self)
        
    @property
    def pos(self):
        return [self._x, self._y]
        
    @pos.setter
    def pos(self, value):
        self._x = value[0]
        self._y = value[1]
        
class Centroid(Unit):

    def __init__(self, x, y):
        self._data_sum = [0, 0]
        self._data_count = 0
        super().__init__(x, y)
        
    def add_datum(self, x, y):
        self._data_sum[0] += x
        self._data_sum[1] += y
        self._data_count += 1
        
    def update(self):
        if self._data_count > 0:
            self._x = self._data_sum[0]/self._data_count
            self._y = self._data_sum[1]/self._data_count
            self._data_sum = [0, 0]
            self._data_count = 0

class Neuron(Unit):
        
    def set_rates(self, epsilon, lamb):
        self._lambda = lamb
        self._epsilon = epsilon
        
class NeuronNG(Neuron):
      
    def pre_count(self, n):
        self._coefficient = [self._epsilon * np.e**(-i/self._lambda) for i in range(n)]
    
    def update(self, i, x, y):
        self._x += self._coefficient[i] * (x - self._x)
        self._y += self._coefficient[i] * (y - self._y)
        
class NeuronKHN(Neuron):

    def set_function(self, g):
        self._g = g
        
    def update(self, w_0, x, y):
        Gd = self._g(self.distance(w_0[0], w_0[1]), l=self._lambda)
        self._x += self._epsilon * Gd * (x - self._x)
        self._y += self._epsilon * Gd * (y - self._y)

