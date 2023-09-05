import csv
import tikzplotlib
import matplotlib.pyplot as plt
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


    def MC_step(self):
        # Anna
        T = self.T
        L = self.L
        # steps 1-3 p.12

        c = [1,3] # tuple of the chosen coordinate; here e.g. set to (1,3)
        s_old = self.s[tuple(c)]
        s_new = 2 # set this to the new value of s[c]
        neighbours = np.array([[0,1],[1,0],[-1,0],[0,-1]])
        neighbours = np.mod(c+neighbours, np.array([L,L]))
        s_neighbours = self.s[tuple(map(tuple,neighbours.T))]
        delta_E = -self.J*(np.sum(s_new == s_neighbours)-np.sum(s_old == s_neighbours))

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

    def plot_state(self, show_plt=True, filename=None):
        # Theo
        # plt.style.use('_mpl-gallery-nogrid')
        fig, ax = plt.subplots()
        ax.imshow(self.s, cmap='Set1')
        ax.set(xlabel=r'x', ylabel=r'y')

        if filename:
            tikzplotlib.save(filename)

        if show_plt:
            plt.show()




def plot_energies(self):
    # Carmen
    # taking averages, etc.
    pass
    


if __name__ == '__main__':
    # main starts here
    # designing experiments: Anna, Theo, Carmen


    if 1:
        # Test the function MC_step
        model = Potts(4)
        model.MC_step()


