import csv
import tikzplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import time

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
    def __init__(self, L=2, T=1, q=2, J=1, M=100):
        # parameters
        self.L = L #number of lattice sites per side
        self.N = L * L #TOTAL number of lattice sites
        self.T = T # temperature
        self.q = q #number of different spin values, integer >=2
        self.J = J
        self.s =  np.random.randint(1, q+1, (L,L), int) #initial state, 2D-matrix of spin states, reading order
        self.E = np.empty(0) # list of energies
        self.M = M # Number of simulation runs

    def MC_step(self):
        # Anna
        T = self.T
        L = self.L
        # steps 1-3 p.12

        # step 1: choose random spin
        # pick random coordinates
        rng = np.random.default_rng()
        cx = rng.choice(self.s.shape[0])      ## UNIFORMLY CHOOSEN? (I would say yes)
        cy = rng.choice(self.s.shape[1])
        c = [cx,cy]

        # step 2: propose state and calculate enrgy change
        s_new = 1+rng.choice(self.q)
        s_old = self.s[tuple(c)]

        neighbours = np.array([[0,1],[1,0],[-1,0],[0,-1]])
        neighbours = np.mod(c+neighbours, np.array([L,L]))
        s_neighbours = self.s[tuple(map(tuple,neighbours.T))]

        delta_E = -self.J*(np.sum(s_new == s_neighbours)-np.sum(s_old == s_neighbours))         # What is the sum doing?

        # Accept or deny change 
        if rd.random() < np.exp(-delta_E/ T):
            self.s[tuple(c)] = s_new

    def run_simulation(self):
        ax = plt.subplot()
        plt.ion()
        for i in range(self.M):
            self.MC_step()
            print(self.s)
            if not i % 10:
                self.plot_state(ax=ax)
                #time.sleep(1)
                plt.pause(0.0001)

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

    def plot_state(self, show_plt=True, filename=None, ax=None):
        # Theo
        # plt.style.use('_mpl-gallery-nogrid')
        if not ax:
            fig, ax = plt.subplot()
        ax.imshow(self.s, cmap='Set1')

        if filename:
            tikzplotlib.save(filename)

        if show_plt:
            plt.show()

# Wann pass, wann kein pass


def plot_energies(self):
    # Carmen
    # taking averages, etc.
    pass
    


if __name__ == '__main__':
    # main starts here
    # designing experiments: Anna, Theo, Carmen


    if 1:
        # Test the function MC_step
        model = Potts(24, q=10, M=1000)
        model.run_simulation()
        print(model.s)
