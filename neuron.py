import numpy as np
import math

class Unit:

    def __init__(self, coords):
        self._coords = np.array(coords)
        
    def distance(self, point):
        return np.linalg.norm(self._coords-point)
        
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
        self._data_sum = np.array(0)
        self._data_count = 0
        super().__init__(coords)
        
    def add_datum(self, point):
        self._data_sum = self._data_sum + point
        self._data_count += 1
        
    def update(self):
        if self._data_count > 0:
            self._coords = self._data_sum/self._data_count
            self._data_sum = np.array(0)
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
        self._coords += self._coefficient[k] * (point - self._coords)
        
class NeuronKHN(Neuron):

    @staticmethod
    def set_function(g):
        NeuronKHN._g = g
        
    def update(self, w_0, point):
        EGd = self._epsilon * NeuronKHN._g(self.distance(w_0), l=self._lambda)
        self._coords += EGd * (point - self._coords)

