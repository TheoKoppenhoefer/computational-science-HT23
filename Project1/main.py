import csv
import tikzplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pandas as pd
from itertools import count
import numba as nb
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
    def __init__(self, L=2, T=1, q=2, J=1, cs=False):
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
        
        
        # calculate the TOTAL energy
        #left and right comparisons, "2 * ..." accounts for the periodic boundary conditions
        lr = np.sum((self.s[:,0:-1] == self.s[:,1:]).astype(int)) + 2 * np.sum((self.s[:,0] == self.s[:,-1]).astype(int))
        #top and bottom comparisons, "2 * ..." accounts for the periodic boundary conditions
        tb = np.sum((self.s[0:-1,:] == self.s[1:,:]).astype(int)) + 2 * np.sum((self.s[0,:] == self.s[-1,:]).astype(int))
        self.e = -self.J * (lr + tb) # total energy with with adding delta_E each time
        # list of energies, one TOTAL energy of the system per iteration, NOT delta_E !
        self.E = [self.e]

        self.neighbours = np.empty((L,L), dtype=np.dtype('(2,4)int'))
        for c_x in range(L):
            for c_y in range(L):
                neighbourhood = np.array([[0,1],[1,0],[-1,0],[0,-1]])
                neighbourhood = np.mod(np.array([c_x,c_y])+neighbourhood, np.array([L,L]))
                self.neighbours[c_x, c_y] = neighbourhood.T

        
        self.rng = np.random.default_rng()

    def MC_step(self):
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
            self.e += delta_E
  
        self.E += [self.e] # append energy to the list

    def run_simulation_fast(self, M=100, M_sampling=5000):
        run_simulation_fast(self.s, self.neighbours, self.J, self.e, self.E, self.L, self.q, self.T, M, M_sampling)

    def run_simulation(self, M=100, M_sampling=5000, show_state=[]):
        """
        M: number of simulation steps. If M<0 then run until energy flattens off
        """
        if show_state:
            #plots
            fig, ax = plt.subplots()
            plt.ion()
        # determine time when the energy plateaus off by taking two moving averages
        # ma_1 and ma_2 of length -M of the energies and determining when ma_1<=ma_2
        ma_1 = 0
        ma_2 = 0
        t_end = M if M>=0 else np.inf

        for i in count(0):
            self.e = MC_step_fast(self.s, self.neighbours, 
                                          self.J, self.e, self.L, self.q, self.T)
            self.E += [self.e] # append energy to the list

            # compute the moving averages of the energy
            if M<0:
                if i+2*M >= 0:
                    ma_1 -= self.E[i+2*M]
                if i+M >= 0:
                    ma_1 += self.E[i+M]
                    ma_2 -= self.E[i+M]
                ma_2 += self.E[i]
            if -2*M<i and t_end==np.inf and ma_1 <= ma_2:
                t_end = i+M_sampling

            #print(self.s)
            #if i % 100 == 0:
                # TODO: get_E total energy comparison with total enery calculated in marcov step
            
            if i in show_state:
                self.plot_state(ax=ax)
                plt.pause(0.0001)

            if i >= t_end:
                break
        plt.ioff()

    def get_E(self, s, J_p):
        # Carmen
        # calculate the energy
        #left and right comparisons, "2 * ..." accounts for the periodic boundary conditions
        lr = np.sum((s[:,0:-1] == s[:,1:]).astype(int)) + 2 * np.sum((s[:,0] == s[:,-1]).astype(int))
        #top and bottom comparisons, "2 * ..." accounts for the periodic boundary conditions
        tb = np.sum((s[0:-1,:] == s[1:,:]).astype(int)) + 2 * np.sum((s[0,:] == s[-1,:]).astype(int)) 
        
        return -J_p * (lr + tb)
    
    def get_stats(self):
        # return mean and variance
        t_0 = analyse_energy(self.E)
        return np.mean(self.E[t_0:]), np.var(self.E[t_0:])

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


@nb.njit()
def run_simulation_fast(s, neighbours, J, e, E, L, q, T, M=100, M_sampling=5000):
    """
    M: number of simulation steps. If M<0 then run until energy flattens off
    """
    # determine time when the energy plateaus off by taking two moving averages
    # ma_1 and ma_2 of length -M of the energies and determining when ma_1<=ma_2
    ma_1 = 0
    ma_2 = 0
    t_end = M if M>=0 else np.inf

    i = 0
    while True:
        e = MC_step_fast(s, neighbours, 
                                        J, e, L, q, T)
        E += [e] # append energy to the list

        # compute the moving averages of the energy
        if M<0:
            if i+2*M >= 0:
                ma_1 -= E[i+2*M]
            if i+M >= 0:
                ma_1 += E[i+M]
                ma_2 -= E[i+M]
            ma_2 += E[i]
        if -2*M<i and t_end==np.inf and ma_1 <= ma_2:
            t_end = i+M_sampling

        #print(s)
        #if i % 100 == 0:
            # TODO: get_E total energy comparison with total enery calculated in marcov step
        if i >= t_end:
            break
        i += 1

@nb.njit()
def MC_step_fast(s, neighbours, J, e, L, q, T):
    # Anna
    # steps 1-3 p.12

    # step 1: choose random spin
    # pick random coordinates
    c = (np.random.randint(L),np.random.randint(L))

    # step 2: propose state and calculate enrgy change
    s_new = 1+np.random.randint(q)
    s_old = s[c]
    s_neighbours = np.empty((4,2))
    for i, neighbour in enumerate(neighbours[c].T):
        s_neighbours[i,:] = s[neighbour[0],neighbour[1]]
    delta_E = -J*(np.sum(s_new == s_neighbours)-np.sum(s_old == s_neighbours))         # What is the sum doing?

    # Accept or deny change 
    if np.random.random() < np.exp(-delta_E/T):
        s[c] = s_new
        # calculate new total energy from older energy 
        e += delta_E
    return e

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
    

def analyse_energy(E, n=1000):
    # determine time t_0 where the energy plateaus off by taking two moving averages
    # ma_1 and ma_2 and determining when ma_1<ma_2
    M = len(E)
    t_0 = M
    n2 = 2*n
    ma_1 = np.sum(E[0:n]) # calculate 100*moving average of first 100 terms
    ma_2 = np.sum(E[n:n2])
    for i in range(M-n2):
        if ma_1 <= ma_2:
            return i
        ma_1 -= E[i]
        ma_1 += E[i+n]
        ma_2 -= E[i+n]
        ma_2 += E[i+n2]

    # Notify if E does not plateau off
    if t_0 == M:
        print(f'in analyse_energy(): t_0 could not be determined.')
    return np.inf


    


if __name__ == '__main__':
    # main starts here
    # designing experiments: Anna, Theo, Carmen
        
    # TODO: compare hot start - cold start final results
    
    if False:
        # Create a time series of the temperature
        model = Potts(20, q=10, M=1000)
        model.run_simulation()
        model.write_E(filename='Data/Energy_step_M1000_L20_q10.csv')
    
    if False:
        # Show a nice plot for high temperature
        model = Potts(10, T=1E5, q=5, M=10000)
        model.run_simulation(show_state=range(1,10000,200))
    
    if False:
        # and for low temperature
        model = Potts(10, T=1E-5, q=5, M=10000)
        model.run_simulation(show_state=range(1,10000,200))


    # Define the parameters for the experiments
    qs = [2,10]# range(2,10,3)
    Ts = np.linspace(1E-2,2,30)
    M = -1000
    M_sampling = 5000
    L = 300

    means = pd.DataFrame(columns=Ts, index=qs)
    variances = pd.DataFrame(columns=Ts, index=qs)

    if True:
        # Run the simulation for various T and q
        for q in qs:
            for T in Ts:
                model = Potts(L, T, q)
                print(f'running model for L={L}, T={T}, q={q}')
                t1 = time.perf_counter()
                model.run_simulation_fast(M, M_sampling)
                print(f'it took {time.perf_counter()-t1}.')
                # model.write_E(filename=f'Data/Energy_step_L{L}_T{T}_q{q}.csv')
                means.loc[q][T], variances.loc[q][T] = model.get_stats()
    means.to_pickle(f'Data/means_L{L}.pkl')
    variances.to_pickle(f'Data/variances_L{L}.pkl')
    
    if True:
        """
        # analyse E for various T and q
        for q in qs:
            for T in Ts:
                # load the list of energies from the file
                E = np.loadtxt(f'Data/Energy_step_L{L}_T{T}_q{q}.csv', delimiter=',')

                # get the time t_0 when the energy plateaus off
                t_0 = analyse_energy(E)
                # t_0 = len(E)-M_sampling

                # return the mean and variance
                means.loc[q][T] = np.mean(E[t_0:])
                variances.loc[q][T] = np.var(E[t_0:])
        """
                

        means.read_pickle(f'Data/means_L{L}.pkl')
        variances.read_pickle(f'Data/variances_L{L}.pkl')
        # plot results nicely
        fig, ax = plt.subplots()
        for q in qs:
            # plot the values in dependence of the temperature
            ax.errorbar(Ts, means.loc[q], yerr=variances.loc[q], label=f'{q}')
        ax.legend(title='parameter $q$', labels=qs)
        ax.set_xlabel('temperature $T$')
        ax.set_ylabel('energy $E$')
        plt.show()
