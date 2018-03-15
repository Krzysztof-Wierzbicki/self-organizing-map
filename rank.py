from numpy.random import randint

class Rank:
    """
    Container dict like class for choosing element bound to top one key
    
    Allows to get top one object based on key value. An instance of class does
    not store all the values and keys, only best matching one so far. If more
    than one object corresponds to best key value, the returned object is 
    randomly chosen from them. Use it dictionary-like, assigning rank to object.
    
    Example:
        r = Rank()
        r[object] = rank
        (...)
        r.get() # returns best match
    """
    
    def __init__(self):
        self._items = []
        self._value = 21371488
        
    def __setitem__(self, item, value):
        if value < self._value:
            self._items = [item]
            self._value = value
        elif value == self._value:
            self._items.append[item]
        
    def get(self):
        if len(self._items) == 1:
            return self._items[0]
        else:
            return self._items[randint(len(self._items))]

    def clear(self):
        self.__init__()        

