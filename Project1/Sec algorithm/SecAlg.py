import csv
import tikzplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pandas as pd
from itertools import count
import numba as nb
import time
from scipy.stats import maxwell

from pathlib import Path
#pathname_data = Path("C:/Users/annar/OneDrive - Lund University/Lund Studium/Mathematikstudium/Third Semester/IntCompSience/computational-science-HT23/Project1/computational-science-HT23/Project1/Sec algorithm/Data")
#pathname_gen = Path("C:/Users/annar/OneDrive - Lund University/Lund Studium/Mathematikstudium/Third Semester/IntCompSience/computational-science-HT23/Project1/computational-science-HT23/Project1/Sec algorithm")
#pathname_plots = Path("C:/Users/annar/OneDrive - Lund University/Lund Studium/Mathematikstudium/Third Semester/IntCompSience/computational-science-HT23/Project1/computational-science-HT23/Project1/Plots")

pathname_data = Path("Data")
pathname_gen = Path("")
pathname_plots = Path("Plots")

#plt.style.use('rc.mplstyle')
plt.style.use(pathname_gen/'rc.mplstyle')

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
            self.s = np.random.randint(1, q+1, (L,L))
        
        # calculate the TOTAL energy
        self.e = get_E(self.s, J)
        # list of energies, one TOTAL energy of the system per iteration, NOT delta_E !
        self.E = [self.e]
        
        self.neighbours = np.empty((L,L), dtype=np.dtype('(2,4)int'))
        self.neighbours = initialise_neighbours_fast(L, self.neighbours)
        
        self.rng = np.random.default_rng()

    def run_simulation_fast(self, M=100, M_sampling=5000):
        run_simulation_fast(self.s, self.neighbours, self.J, self.e, self.E, self.L, self.q, self.T, M, M_sampling)

    def run_simulation(self, M=100, M_sampling=5000, show_state=[], save_state=[], filename=''):
        """
        M: number of simulation steps. If M<0 then run until energy flattens off
        M_sampling: number of steps to take after equilibrium was reached
        show_state: frames in which to plot the state
        save_state: frames in which to save the state
        filename: location to store the states
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
            self.e = Gibbs_step(self.s, self.neighbours, 
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
                self.plot_state(True, ax, i)
                plt.pause(0.03)
            if i in save_state:
                self.plot_state(frame_nbr=i, filename=pathname_data/f'{filename}_{i}.')
            if i >= t_end:
                break
        plt.ioff()
    
    def get_stats(self, M_sampling=0):
        # return mean and variance
        t_0 = len(self.E)-M_sampling if M_sampling else analyse_energy(self.E)
        return np.mean(self.E[t_0:]), np.var(self.E[t_0:]), t_0

    def write_E(self, filename='Data/Energies.csv'):
        # write self.E to a file
        # Theo
        with open(filename, 'w') as f:
            wr = csv.writer(f)
            wr.writerow(self.E)

    def plot_state(self, show_plt=False, ax=None, frame_nbr=None, filename=None):
        # Theo
        if not ax: ax = plt.subplot()
        ax.clear()
        ax.imshow(self.s, cmap='Set1')
        if frame_nbr: ax.set_title(f"frame {frame_nbr}")

        if filename:
            tikzplotlib.save(filename)

        if show_plt:
            plt.show()

@nb.njit()
def initialise_neighbours_fast(L, neighbours):
    for c_x in range(L):
        for c_y in range(L):
            neighbourhood = np.array([[0,1],[1,0],[-1,0],[0,-1]])
            neighbourhood = np.mod(np.array([c_x,c_y])+neighbourhood, np.array([L,L]))
            neighbours[c_x, c_y] = neighbourhood.T
    return neighbours

#@nb.njit()
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
        e = Gibbs_step(s, neighbours, 
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

        #comparing get_E with e
        if i % 50 == 0:
            getE = get_E(s, J)
            if getE != e:
                print('get_E: ', getE, 'e: ', e)
                
        if i >= t_end:
            break
        i += 1


@nb.njit()
def get_E(s, J):
    # Carmen
    # calculate the energy
    #left and right comparisons
    lr = np.sum(s[:,0:-1] == s[:,1:]) + np.sum(s[:,0] == s[:,-1])
    #top and bottom comparisons
    tb = np.sum(s[0:-1,:] == s[1:,:]) + np.sum(s[0,:] == s[-1,:]) 
    return -J * (lr + tb)

################MC step

@nb.njit()    
def s_neighbours_fun(s, neighbours,c):
    s_neighbours = np.empty(4)
    for i, neighbour in enumerate(neighbours[c].T):
        s_neighbours[i] = s[neighbour[0],neighbour[1]]
    return s_neighbours

@nb.njit()   
def Props_fun(s_neighbours, q, T):

    Props = np.empty(q)
    for i in np.arange(1,q+1):
        Props[i-1] = np.exp(1/T*np.sum(i == s_neighbours))
    Props = 1/(np.sum(Props))*Props
    Props[np.isnan(Props)] = 0
    return Props

@nb.njit()   
def energy(s_neighbours, J, e, s_new, s_old):
    delta_E = -J*(np.sum(s_new == s_neighbours)-np.sum(s_old == s_neighbours))         
    e += delta_E
    return e

#@nb.njit()
def Gibbs_step(s, neighbours, J, e, L, q, T):
    
    # choose random spin
    c = (np.random.randint(L),np.random.randint(L))
    s_old = s[c] 

    # find neighbours and calculate Propabilities
    s_neighbours = s_neighbours_fun(s, neighbours, c)
    Props = Props_fun(s_neighbours, q, T)
    #print(Props)
    s_new =  np.random.choice(np.arange(1, q+1), p = Props)
    s[c] = s_new

    #energy
    e = energy(s_neighbours, J, e, s_new, s_old)
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
    
    ax.figure.savefig('EnergyVsInterations.png')

def plot_energies_distr(E, show_plt=True, filename=None, fit_maxwell=False):
    plt.style.use(pathname_gen/'rc.mplstyle')
    fig, ax = plt.subplots()
    ax.hist(E, bins=150, density=True, label='Data')
    ax.set_xlabel('Energy $E$')
    ax.set_ylabel('Share of states')
    if fit_maxwell:
        # fit a maxwell distribution to the data
        params = maxwell.fit(E[::100], loc=min(E))
        x = np.linspace(min(E), max(E), 1000)
        ax.plot(x, maxwell.pdf(x, *params), label='Maxwell distribution')
    if filename: tikzplotlib.save(f'{filename}.pgf')
    ax.set_title('Distribution of the energy in equilibrium')
    ax.legend()
    if show_plt: plt.show()

def plot_energies_t0(E, t_0, show_plt=True, filename=None):
    fig, ax = plt.subplots()
    ax.plot(E)
    ax.axvline(t_0, label='$t_0$')
    ax.set_xlabel('Iteration $i$')
    ax.set_ylabel('Energy $E$')
    if filename: tikzplotlib.save(filename)
    ax.set_title('Energy evolution')
    if show_plt: plt.show()

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
    # Create a time series of the temperature with Bolzmann
    M = -5000
    M_sampling = 10 # int(1E6)
    filename = pathname_data/'Energies_Boltzmann_Distribution.csv'
    if True:
        # run the simulation
        model = Potts(300, q=10, T=1E2)
        model.run_simulation(M, M_sampling)
        model.write_E(filename)
    if True:
        # and plot it
        E = np.loadtxt(filename, delimiter=',')
        t_0 = len(E)-M_sampling
        plot_energies_distr(E[t_0:], filename= pathname_plots/'Energies_Boltzmann_Distribution', fit_maxwell=True)
        # plot_energies_t0(E, t_0)

    
    if True:
        # Show a nice animation for high temperature
        model = Potts(20, T=1E5, q=5)
        model.run_simulation(10000, show_state=range(0,10000,200), save_state=[10000], filename=pathname_plots/'High_temp_state')
        # E = model.E
        # plot_energies_t0(E, 0)
    
    if True:
        # and for low temperature
        model = Potts(20, T=1E-1, q=5)
        model.run_simulation(10000, show_state=range(0,10000,200))


    # Define the parameters for the experiments
    qs = [2,10]# range(2,10,3)
    Ts = np.linspace(1E-1,2,10)
    M = -1000
    M_sampling = 1#5000
    L = 10 #500

    means = pd.DataFrame(columns=Ts, index=qs)
    variances = pd.DataFrame(columns=Ts, index=qs)
    t_0s = pd.DataFrame(columns=Ts, index=qs) # time it takes to reach equilibrium

    if True:
        # Run the simulation for various T and q
        for q in qs:
            for T in Ts:
                print(f'running model for L={L}, T={T}, q={q}')
                # pf = time.perf_counter()
                model = Potts(L, T, q)
                # print(f'setup {time.perf_counter()-pf}.')
                # pf = time.perf_counter()
                model.run_simulation_fast(M, M_sampling)
                # print(f'running {time.perf_counter()-pf}.')
                means.loc[q][T], variances.loc[q][T], t_0s.loc[q][T] = model.get_stats(M_sampling)
        means.to_pickle(pathname_data/f'means_L{L}.pkl')
        variances.to_pickle(pathname_data/f'variances_L{L}.pkl')
        t_0s.to_pickle(pathname_data/f't0s_L{L}.pkl')
    
    if True:
        # plot variances and means
        means = pd.read_pickle(pathname_data/f'means_L{L}.pkl')
        variances = pd.read_pickle(pathname_data/f'variances_L{L}.pkl')
        fig, ax = plt.subplots()
        for q in qs:
            # plot the values in dependence of the temperature
            ax.errorbar(Ts, means.loc[q], yerr=variances.loc[q], label=f'{q}')
        ax.legend(title='parameter $q$', labels=qs)
        ax.set_xlabel('Eemperature $T$')
        ax.set_ylabel('Energy $E$')
        plt.show()

        # plot t_0s 
        t_0s = pd.read_pickle(pathname_data/f't0s_L{L}.pkl')

        fig, ax = plt.subplots()
        for q in qs:
            ax.plot(t_0s.loc[q], label=f'{q}')
        ax.legend(title='Parameter $q$', labels=qs)
        ax.set_xlabel('Temperature $T$')
        ax.set_ylabel('Time $t_0$')
        plt.show()