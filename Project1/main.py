import csv
import tikzplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pandas as pd

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
    def __init__(self, L=2, T=1, q=2, M=100, J=1, cs=False):
        # parameters
        self.L = L #number of lattice sites per side
        self.N = L * L #TOTAL number of lattice sites
        self.T = T # temperature
        #TODO: check that the user gives a valid q value
        self.q = q #number of different spin values, integer >=2
        
        self.J = J
        
        self.s = np.empty((L,L)) #spin state of the system
        #TODO: check that cs makes sense with respect to q
        if cs != False: #cold start
            self.s.fill(cs) 
        else: #hot start
            self.s = np.random.randint(1, q+1, (L,L), int) 
        
        self.M = int(M) # Number of simulation runs
        self.E = np.empty(self.M) # list of energies, one TOTAL energy of the system per iteration, NOT delta_E !
        
        # calculate the TOTAL energy
        #left and right comparisons, "2 * ..." accounts for the periodic boundary conditions
        lr = np.sum((self.s[:,0:-1] == self.s[:,1:]).astype(int)) + 2 * np.sum((self.s[:,0] == self.s[:,-1]).astype(int))
        #top and bottom comparisons, "2 * ..." accounts for the periodic boundary conditions
        tb = np.sum((self.s[0:-1,:] == self.s[1:,:]).astype(int)) + 2 * np.sum((self.s[0,:] == self.s[-1,:]).astype(int))
        self.e = -self.J * (lr + tb) # total energy with with adding delta_E each time
        self.E[0] = self.e

        self.neighbours = np.empty((L,L), dtype=np.dtype('(2,4)int'))
        for c_x in range(L):
            for c_y in range(L):
                neighbourhood = np.array([[0,1],[1,0],[-1,0],[0,-1]])
                neighbourhood = np.mod(np.array([c_x,c_y])+neighbourhood, np.array([L,L]))
                self.neighbours[c_x, c_y] = neighbourhood.T

        
        self.rng = np.random.default_rng()


    def MC_step(self, step):
        # Anna
        # steps 1-3 p.12

        # step 1: choose random spin
        # pick random coordinates
        c = tuple(self.rng.choice(self.L, size=2))

        # step 2: propose state and calculate enrgy change
        s_new = 1+self.rng.choice(self.q)
        s_old = self.s[c]

        s_neighbours = self.s[tuple(map(tuple,self.neighbours[c]))]

        delta_E = -self.J*(np.sum(s_new == s_neighbours)-np.sum(s_old == s_neighbours))         # What is the sum doing?

        # Accept or deny change 
        if self.rng.random() < np.exp(-delta_E/self.T):
            self.s[c] = s_new
            # calculate new total energy from older energy 
            self.e = self.e + delta_E
  
        self.E[step] = self.e

    def run_simulation(self, show_state=[]):

        #plots
        ax = plt.subplot()
        plt.ion()
        for i in range(self.M):
            self.MC_step(i)
            #print(self.s)
            #if i % 100 == 0:
                # TODO: get_E total energy comparison with total enery calculated in marcov step
            if i in show_state:
                self.plot_state(ax=ax)
                plt.pause(0.0001)
        plt.ioff()

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
            ax = plt.subplot()
        ax.imshow(self.s, cmap='Set1')

        if filename:
            tikzplotlib.save(filename)

        if show_plt:
            plt.show()


def plot_energies(filename, show_plt=True): #dont understand the ax thing in plot_state
    # Carmen
    # read filename (total energy per iteration) and plot it 
    data = pd.read_csv(filename, header=None).T
    ax = data.plot(legend=False, figsize = (8,7), fontsize=12)
    ax.set_xlabel('Iterations', fontsize=15)
    ax.set_ylabel('Energy', fontsize=15)
    ax.set_title('Total Energy vs Iterations', fontsize=15)
    
    if show_plt:
        plt.show()
    
    ax.figure.savefig(filename[0:-3] + 'png')
    
    
def analyse_energy(E):
    # determine time t_0 where the energy plateaus off by taking two moving averages
    # ma_1 and ma_2 and determining when ma_1<ma_2
    M = len(E)
    t_0 = M
    ma_1 = np.sum(E[0:100]) # calculate 100*moving average of first 100 terms
    ma_2 = np.sum(E[100:200])
    for i in range(M-200):
        if ma_1 <= ma_2:
            t_0 = i
            break
        ma_1 -= E[i]
        ma_1 += E[i+100]
        ma_2 -= E[i+100]
        ma_2 += E[i+200]

    # Notify if E does not plateau off
    if t_0 == M:
        print(f'in analyse_energy(): t_0 could not be determined.')
    
    # return the mean and variance
    return np.mean(E[t_0:]), np.var(E[t_0:])


    


if __name__ == '__main__':
    # main starts here
    # designing experiments: Anna, Theo, Carmen
        
    # TODO: compare hot start - cold start final results
    
    if False:
        # Create a time series of the temperature
        model = Potts(20, q=10, M=1000)
        model.run_simulation()
        model.write_E(filename='Data/Energy_step_M1000_L20_q10.csv')
    
    if True:
        # Show a nice plot for high temperature
        model = Potts(10, T=1E5, q=5, M=10)
        model.run_simulation(show_state=range(1,10,200))
    
    if True:
        # and for low temperature
        model = Potts(10, T=1E-5, q=5, M=10)
        model.run_simulation(show_state=range(1,10,200))


    # Define the parameters for the experiments
    qs = range(2,10,5)
    Ts = np.linspace(1E-2,1E2,10)
    M = int(1E4)
    L = 10

    if True:
        # Run the simulation for various T and q
        for q in qs:
            for T in Ts:
                model = Potts(L, T, q, M)
                model.run_simulation()
                model.write_E(filename=f'Data/Energy_step_L{L}_T{T}_q{q}_M{M}.csv')

    means = pd.DataFrame(columns=Ts, index=qs)
    variances = pd.DataFrame(columns=Ts, index=qs)
    # convert this to dataframes
    
    if True:
        # analyse E for various T and q
        for q in qs:
            for T in Ts:
                # load the list of energies from the file
                E = np.loadtxt(f'Data/Energy_step_L{L}_T{T}_q{q}_M{M}.csv', delimiter=',')
                means.loc[q][T], variances.loc[q][T] = analyse_energy(E)
        
        # TODO: print means, variances to file
        # TODO: read means, variances from file

        # plot results nicely
        ax = plt.subplot()
        for q in qs:
            # plot the values in dependence of the temperature
            ax.errorbar(Ts, means.loc[q], yerr=variances.loc[q], label=f'{q}')
        ax.legend(title='parameter $q$', labels=qs)
        ax.set_xlabel('temperature $T$')
        ax.set_ylabel('energy $E$')
        plt.show()
