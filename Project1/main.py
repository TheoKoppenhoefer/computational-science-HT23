import numpy as np
import random as rd

# good programming practice in python
# - avoid loops (vectorise, use numpy, stencils)
# - avoid function calls


# Extensions:
# - magnetisation


class Irsing():
    # Carmen: class structure
    def __init__(self):
        # parameters
        self.N
        self.L
        self.T
        self.q
        self.J_p = 1
        self.s # initialise this somehow
        # s = [[1,2,1],[1,3,1],...] 2D-matrix os spin states, reading order
        self.E # list of energies


    def MC_step(self, s):
        T = self.T
        # steps 1-3 p.12

    def nearest_neighbours():
        # returns a stencil given a coordinate

    def run_simulation(self):
        # initialise
        
        for i in range():

    def get_E(self):
        # calculate the energy

    def write_E():

    def plot_state(self):



def plot_energies(self):
    


if __name__ == '__main__':
    # main starts here


