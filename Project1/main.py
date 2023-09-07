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
    def __init__(self, L=2, T=1, q=2, J=1, M=100, e =0):
        # parameters
        self.L = L #number of lattice sites per side
        self.N = L * L #TOTAL number of lattice sites
        self.T = T # temperature
        self.q = q #number of different spin values, integer >=2
        self.J = J
        self.s =  np.random.randint(1, q+1, (L,L), int) #initial state, 2D-matrix of spin states, reading order
        self.E = np.empty(0) # list of energies
        self.M = M # Number of simulation runs
        self.e = e # total energy with with adding delta_E each time


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
            # calculate new total energy from older energy 
            self.e = self.e + delta_E
  
        self.E = np.append(self.E, self.e)

    def run_simulation(self, show_state=[]):

        # Calculate total Energy
        self.E = np.append(self.E,self.e)

        #plots
        ax = plt.subplot()
        plt.ion()
        for i in range(self.M):
            self.MC_step()
            #print(self.s)
            #if i % 100 == 0:
                # get_E total energy comparison with total enery calculated in marcov step
            if i in show_state:
                self.plot_state(ax=ax)
                plt.pause(0.0001)

    def get_E(self, s, J_p):
        # Carmen
        # calculate the energy
        #left and right comparisons, "2 * ..." accounts for the periodic boundary conditions
        lr = np.sum((s[:,0:-1] == s[:,1:]).astype(int)) + 2 * np.sum((s[:,0] == s[:,-1]).astype(int))
        #top and bottom comparisons, "2 * ..." accounts for the periodic boundary conditions
        tb = np.sum((s[0:-1,:] == s[1:,:]).astype(int)) + 2 * np.sum((s[0,:] == s[-1,:]).astype(int)) 
        
        return -J_p * (lr + tb)

    def write_E(self, filename='Data/Energies.csv'):
        # write self.E to a file
        # Theo
        with open(filename, 'w') as f:
            wr = csv.writer(f)
            wr.writerow(self.E)

    def plot_state(self, show_plt=True, filename=None, ax=None):
        # Theo
        if not ax:
            fig, ax = plt.subplot()
        ax.imshow(self.s, cmap='Set1')

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

    if False:
        # Create a time series of the temperature
        model = Potts(20, q=10, M=1000)
        model.run_simulation()
        model.write_E(filename='Data/Energy_step_M1000_L20_q10.csv')


    if False:
        # Show a nice plot for high temperature
        model = Potts(10, T=1E5, q=5, M=10000)
        model.run_simulation(show_state=range(1,10000,200))

    if True:
        # and for low temperature
        model = Potts(10, T=1E-5, q=5, M=10000)
        model.run_simulation(show_state=range(1,10000,200))
