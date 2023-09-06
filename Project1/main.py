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
    def __init__(self, L=2, T=1, q=2, J=1):
        # parameters
        self.L = L #number of lattice sites per side
        self.N = L * L #TOTAL number of lattice sites
        self.T = T #temperature
        self.q = q #number of different spin values, integer >=2
        self.J = J
        self.s =  np.random.randint(1, q+1, (L,L), int) #initial state, 2D-matrix of spin states, reading order
        self.E = np.empty(0) # list of energies
        self.M = 100 # Number of simulation runs

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

        print('s_new:', s_new, 's_old:', s_old)
        neighbours = np.array([[0,1],[1,0],[-1,0],[0,-1]])
        neighbours = np.mod(c+neighbours, np.array([L,L]))
        print('c:',c ,'neighbours:', neighbours)
        s_neighbours = self.s[tuple(map(tuple,neighbours.T))]
        print('s_neighb:', s_neighbours)

        delta_E = -self.J*(np.sum(s_new == s_neighbours)-np.sum(s_old == s_neighbours))         # What is the sum doing?
        print('delta_E:', delta_E)

        # Accept or deny change 
        if rd.random() < np.exp(-delta_E/ T):
            self.s[tuple(c)] = s_new

    def run_simulation(self):

        # initialise s
        s = self.s
        for i in range(self.M):
            self.MC_step()
            print(self.s)
            if i % 10 == 0:
                self.plot_state()

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

# Wann pass, wann kein pass
# Warum sind alle funktionen in der Klasse (VOrteil)


def plot_energies(self):
    # Carmen
    # taking averages, etc.
    pass
    


if __name__ == '__main__':
    # main starts here
    # designing experiments: Anna, Theo, Carmen


    if 1:
        # Test the function MC_step
        model = Potts(10)
        model.run_simulation()
        print(model.s)
