import argparse 

class Parsing:

    def init_parser(self, a, d, f):
        self._algorithms = list(a)
        self._datasets = list(d)
        self._funcs = list(f)

    def create_parser(self):
        self._parser = argparse.ArgumentParser(description='Self organising map')
        self._parser.add_argument('dataset', nargs='?', default='iris', 
                            help='dataset name', choices=self._datasets)
        self._parser.add_argument('-n', '--neurons', metavar='N', type=int, default=5, 
                            help='set number of central units, default: 5')
        self._parser.add_argument('--no-plot', action='store_false', 
                            help='don\'t plot')
        self._parser.add_argument('--no-errors', action='store_false', 
                            help='don\'t plot errors')
        self._parser.add_argument('-c', '--count', type=int, default=100, 
                            help='set max number of epochs, default: 100')
        self._parser.add_argument('-a', '--algorithm', default='neural-gas', action='store', 
                            help='fitting algorithm', choices=self._algorithms)
        self._parser.add_argument('-f', '--function', default='other', action='store', 
                            help='kohonen function', choices=self._funcs)
        self._parser.add_argument('-l', '--lambda', dest='lamb', default=None, type=float, 
                            help='lambda value, sensible values: 0.0000001 for kohonen other, 0.5 for neural gas and kohonen step')
        self._parser.add_argument('-e', '--epsilon', default=0.01, type=float, 
                            help='epsilon value, the learning rate') 
        
    def parse(self):
        self._args = self._parser.parse_args()
        
    def get_lambda(self):
        lamb = self._args.lamb
        if lamb == None:
            if self._args.algorithm == 'neural-gas':
                lamb = 0.5
            elif self._args.algorithm == 'kohonen':
                if self._args.function == 'other':
                    lamb = 0.0000001
                else:
                    lamb = 0.5
        
        return lamb
        
    def get_dataset(self):
        return self._args.dataset
        
    def get_neurons(self):
        return self._args.neurons
        
    def get_count(self):
        return self._args.count
        
    def get_algorithm(self):
        return self._args.algorithm
        
    def get_function(self):
        return self._args.function
        
    def get_epsilon(self):
        return self._args.epsilon
        
    def get_no_errors(self):
        return self._args.no_errors
        
    def get_no_plot(self):
        return self._args.no_plot

