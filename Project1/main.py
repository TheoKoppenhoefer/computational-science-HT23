import csv
from xml.dom.expatbuilder import parseFragmentString
import numpy as np
import random as rd

# good programming practice in python
# - avoid loops (vectorise, use numpy, stencils)
# - avoid function calls


# Extensions:
# - magnetisation


class Potts:
    """
    Implementation of the Potts model
    """
    # Carmen: class structure
    def __init__(self, L=2):
        # parameters
        self.N = L*L
        self.L = L
        self.T = 1
        self.q = 2
        self.J = 1
        self.s = np.ones((L,L))# initialise this somehow
        # s = [[1,2,1],[1,3,1],...] 2D-matrix os spin states, reading order
        self.E = [] # list of energies


    def MC_step(self, s):
        # Anna
        T = self.T
        # steps 1-3 p.12

    def nearest_neighbours():
        # returns a stencil given a coordinate
        pass

    def run_simulation(self):
        # Anna
        # initialise
        pass

    def get_E(self):
        # Carmen
        # calculate the energy
        pass

    def write_E(self, filename='Data/Energies.csv'):
        # write self.E to a file
        # Theo
        with open(filename, 'w') as f:
            wr = csv.writer(f)
            wr.writerow(self.E)
    
    def write_state(self):
        pass


    def plot_state(self):
        # Theo
        pass



def plot_energies(self):
    # Carmen
    # taking averages, etc.
    pass
    


if __name__ == '__main__':
    # main starts here
    # designing experiments: Anna, Theo, Carmen
    model = Potts(4)
    model.write_E()

