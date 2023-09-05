import numpy as np
#import random as rd

# good programming practice in python
# - avoid loops (vectorise, use numpy, stencils)
# - avoid function calls


# Extensions:
# - magnetisation


class Irsing():
    # Carmen: class structure
    def __init__(self, L, T, q):
        # parameters
        self.L = L #number of lattice sites per side
        self.N = L * L #TOTAL number of lattice sites
        self.T = T #temperature
        self.q = q #number of different spin values, integer >=2
        self.J_p = 1
        self.s =  np.random.randint(1, q+1, (L,L), int) #initial state, 2D-matrix of spin states, reading order
        self.E = np.empty(0) # list of energies

    def MC_step(self, s):
        # Anna
        T = self.T
        # steps 1-3 p.12

    def nearest_neighbours():
        # returns a stencil given a coordinate

    def run_simulation(self):
        # Anna
        # initialise
        
        for i in range():

    def get_E(self):
        # Carmen
        # calculate the energy

    def write_E():
        # Theo

    def plot_state(self):
        # Theo



def plot_energies(self):
    # Carmen
    # taking averages, etc.
    


if __name__ == '__main__':
    # main starts here
    # designing experiments: Anna, Theo, Carmen


