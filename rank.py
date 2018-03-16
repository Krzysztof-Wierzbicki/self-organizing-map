from numpy.random import randint

class Rank(dict):
    """
    Sorted dictionary with generator based iteration over itself and best element chosing
    """
    
    def __iter__(self):
        return [i for i, j in sorted(self.items(), key = lambda pair: pair[1])].__iter__()
        
    def best(self):
        sorted_k = sorted(self.values())
        l = sorted_k.count(sorted_k[0])
        if l == 1:
            return sorted(self.items(), key = lambda pair: pair[1])[0][0]
        else:
            return sorted(self.items(), key = lambda pair: pair[1])[randint(l)][0]

