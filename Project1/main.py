

####### pgf_plot fix outcommented

import csv
import tikzplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pandas as pd
from itertools import count
import numba as nb
from scipy.stats import maxwell
import time
from pathlib import Path
from pgf_plot_fix import tikzplotlib_fix_ncols    

if False:
    pathname = Path("C:/Users/annar/OneDrive - Lund University/Lund Studium/Mathematikstudium/Third Semester/IntCompSience/computational-science-HT23/Project1/computational-science-HT23/Project1/Data")
    pathname_gen = Path("C:/Users/annar/OneDrive - Lund University/Lund Studium/Mathematikstudium/Third Semester/IntCompSience/computational-science-HT23/Project1/computational-science-HT23/Project1")
    pathname_plots = Path("C:/Users/annar/OneDrive - Lund University/Lund Studium/Mathematikstudium/Third Semester/IntCompSience/computational-science-HT23/Project1/computational-science-HT23/Project1/Plots")
else:
    pathname = Path("Data")
    pathname_gen = Path("")
    pathname_plots = Path("Plots")

# good programming practice in python
# - avoid loops (vectorise, use numpy, stencils)
# - avoid function calls

# Extensions:
# - magnetisation

@nb.njit()
def get_E(s, J):
    # Carmen
    # calculate the energy
    #left and right comparisons
    lr = np.sum(s[:,0:-1] == s[:,1:]) + np.sum(s[:,0] == s[:,-1])
    #top and bottom comparisons
    tb = np.sum(s[0:-1,:] == s[1:,:]) + np.sum(s[0,:] == s[-1,:]) 
    return -J * (lr + tb)


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
    s_neighbours = np.empty(4)
    for i, neighbour in enumerate(neighbours[c]):
        s_neighbours[i] = s[neighbour[0],neighbour[1]]
    delta_E = -J*(np.sum(s_new == s_neighbours)-np.sum(s_old == s_neighbours))         # What is the sum doing?

    # Accept or deny change 
    if np.random.random() < np.exp(-delta_E/T):
        s[c] = s_new
        # calculate new total energy from older energy 
        e += delta_E
    return e

@nb.njit()
def Gibbs_step(s, neighbours, J, e, L, q, T):
    
    # choose random spin
    c = (np.random.randint(L),np.random.randint(L))
    s_old = s[c] 

    # find neighbours
    s_neighbours = np.empty(4)
    for i, neighbour in enumerate(neighbours[c]):
        s_neighbours[i] = s[neighbour[0],neighbour[1]]

    # Calculate probabilities
    Props = np.empty(q)
    for i in np.arange(1,q+1):
        Props[i-1] = np.exp(1/T*np.sum(i == s_neighbours))
    Props = 1/(np.sum(Props))*Props
    Props[np.isnan(Props)] = 0
    t = np.random.random()
    
    s_new = 1
    prop_sum =Props[0]
    while prop_sum < t:
        prop_sum += Props[s_new]
        s_new += 1
    s[c] = s_new

    #energy
    delta_E = -J*(np.sum(s_new == s_neighbours)-np.sum(s_old == s_neighbours))         
    e += delta_E
    return e

class Potts:
    """
    Implementation of the Potts model
    """
    # Carmen: class structure
    def __init__(self, L=300, T=1, q=2, J=1, cs=False):
        # parameters
        self.L = L #number of lattice sites per side
        self.N = L * L #total number of lattice sites
        self.T = T # temperature
        self.q = q #number of different spin values, integer >=2
        self.J = J

        self.s = np.empty((L,L)) #spin state of the system
        #TODO: check that cs makes sense with respect to q
        self.cs = cs
        if cs != False: #cold start
            self.s.fill(cs) 
        else: #hot start
            self.s = np.random.randint(1, q+1, (L,L))
        
        # list of total energies of the system
        self.E = np.empty(int(1E8)) # allocate a huge amount of virtual memory for the energies
        self.i = 0 # the current step
        # calculate the total energy
        self.E[0] = get_E(self.s, J)
        self.e = self.E[0]
        
        self.neighbours = np.empty((L,L), dtype=np.dtype('(4,2)int'))
        self.neighbours = initialise_neighbours_fast(L, self.neighbours)
        
        self.rng = np.random.default_rng()

    def run_simulation(self, M=100, M_sampling=5000, show_state=[], save_state=[], filename='', method=MC_step_fast):
        if show_state or save_state or method != MC_step_fast:
            self.run_simulation_slow(M, M_sampling, show_state, save_state, filename, method)
        else:
            self.E, self.e, self.s, self.i = run_simulation_fast(self.s, self.neighbours, self.J, self.e, self.E, self.L, self.q, self.T, self.i, int(M), int(M_sampling), method, self.cs)

    def run_simulation_slow(self, M=100, M_sampling=5000, show_state=[], save_state=[], filename='', method=MC_step_fast):
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
        t_end = M+self.i if M>=0 else np.inf

        while True:
            i = self.i
            self.e = method(self.s, self.neighbours, 
                                          self.J, self.e, self.L, self.q, self.T)
            self.E[self.i+1] = self.e # append energy to the list

            # compute the moving averages of the energy
            if M<0:
                if i+2*M >= 0:
                    ma_1 -= self.E[i+2*M]
                if i+M >= 0:
                    ma_1 += self.E[i+M]
                    ma_2 -= self.E[i+M]
                ma_2 += self.E[i]
            if -2*M<i and t_end==np.inf and ((not self.cs and ma_1 <= ma_2) or (self.cs and ma_1 >= ma_2)):
                t_end = i+M_sampling

            self.i += 1
            if i in show_state:
                self.plot_state(True, ax, i)
                plt.pause(0.003)
            if i in save_state:
                self.plot_state(frame_nbr=i, filename=f'{filename}_{i}')
            if i >= t_end:
                break
        plt.ioff()
    
    def get_stats(self, M_sampling=0):
        # return mean and standard deviation
        t_0 = self.i-M_sampling if M_sampling else analyse_energy(self.E[:self.i])
        return np.mean(self.E[t_0:self.i])/self.N, np.std(self.E[t_0:self.i])/self.N, t_0

    def write_E(self,  filename=pathname_gen/'Energies.csv', max_length=int(1E6), t_0=0):
        # write self.E to a file of maximum length given by max_length
        with open(filename, 'w') as f:
            wr = csv.writer(f)
            wr.writerow(self.E[t_0:self.i:(len(self.E)//max_length+1)]/self.N)

    def plot_state(self, show_plt=False, ax=None, frame_nbr=None, filename=None):
        # Theo
        if not ax: ax = plt.subplot()
        ax.clear()
        ax.imshow(self.s, cmap='Set1')
        if filename: 
            tikzplotlib.save(f'{filename}.pgf')
            ax.figure.savefig(f'{filename}.pdf', bbox_inches='tight')
        if frame_nbr: ax.set_title(f"frame {frame_nbr}")
        if show_plt: plt.show()

    def test_energies(self):
        # check if the energies that the system calculates during the simulation
        # coincide with the actual energy. If not, something went wrong.
        getE = get_E(self.s, self.J)
        if getE != self.e:
            print('Energies do not coincide')
            print('get_E: ', getE, 'e: ', self.e)
        else:
            print('Energies coincide')


@nb.njit()
def initialise_neighbours_fast(L, neighbours):
    for c_x in range(L):
        for c_y in range(L):
            neighbourhood = np.array([[0,1],[1,0],[-1,0],[0,-1]])
            neighbourhood = np.mod(np.array([c_x,c_y])+neighbourhood, np.array([L,L]))
            neighbours[c_x, c_y] = neighbourhood
    return neighbours

@nb.njit()
def run_simulation_fast(s, neighbours, J, e, E, L, q, T, i, M=100, M_sampling=5000, method=MC_step_fast, cs=False):
    """
    M: number of simulation steps. If M<0 then run until energy flattens off
    """
    # determine time when the energy plateaus off by taking two moving averages
    # ma_1 and ma_2 of length -M of the energies and determining when ma_1<=ma_2
    ma_1 = 0
    ma_2 = 0
    t_end = M+i if M>=0 else np.inf

    while True:
        e = method(s, neighbours, J, e, L, q, T)
        E[i+1] += e # append energy to the list

        # compute the moving averages of the energy
        if M<0:
            if i+2*M >= 0:
                ma_1 -= E[i+2*M]
            if i+M >= 0:
                ma_1 += E[i+M]
                ma_2 -= E[i+M]
            ma_2 += E[i]
        if -2*M<i and t_end==np.inf and ((~cs and ma_1 <= ma_2) or (cs and ma_1 >= ma_2)):
            t_end = i+M_sampling

        i += 1
        if i >= t_end:
            break
    return E, e, s, i

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
    
    ax.figure.savefig(filename[0:-3] + 'png', bbox_inches='tight')

def plot_energies_distr(E, show_plt=True, filename=None, fit_maxwell=False):
    # plot a distribution of the energies in E
    plt.style.use(pathname_gen/'rc.mplstyle')
    fig, ax = plt.subplots()
    ax.hist(E, bins=150, density=True, label='Data')
    ax.set_xlabel('Energy $E$ per spin')
    ax.set_ylabel('Share of states')
    if fit_maxwell:
        # fit a maxwell distribution to the data
        params = maxwell.fit(E[::(len(E)//1000)], loc=min(E))
        x = np.linspace(min(E), max(E), 1000)
        ax.plot(x, maxwell.pdf(x, *params), label='Maxwell distribution')
    if filename:
        tikzplotlib.save(f'{filename}.pgf')
        ax.figure.savefig(f'{filename}.pdf', bbox_inches='tight')
    ax.set_title('Distribution of the energy in equilibrium')
    ax.legend()
    if show_plt: plt.show()

def plot_energies_t0(E, t_0=None, show_plt=True, filename=None):
    # plot the energies with the time t_0
    plt.style.use(pathname_gen/'rc.mplstyle')
    fig, ax = plt.subplots()
    ax.plot(E)
    if t_0: ax.axvline(t_0, label='$t_0$')
    ax.set_xlabel('Iteration $i$')
    ax.set_ylabel('Energy $E$ per spin')
    if filename:
        tikzplotlib.save(f'{filename}.pgf')
        ax.figure.savefig(f'{filename}.pdf', bbox_inches='tight')
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
    
    #hot start vs cold start    
    # TODO: fix the plot (energies inverted order?)
    if False: 
        plt.style.use(pathname_gen/'rc.mplstyle')
        Ms = [int(1E6), int(4E4)]
        M_sampling = 0
        Ts = [0.1,1E2]
        q = 2
        methods = [MC_step_fast, Gibbs_step]
        cs = 2
        methodsstring = ['Metropolis', 'Heat - bath']
        L = 100
        N = L*L

        for j, T in enumerate(Ts):
            ax = plt.subplot()
            M = Ms[j]
            for i, method in enumerate(methods):
                hot = Potts(100, q=2, T=T)
                hot.run_simulation(M, M_sampling, method=method)
                ax.plot(hot.E[:hot.i:(hot.i//1000)]/N, label= str(methodsstring[i]) + ', hot ', linewidth=2)
                
                
                cold = Potts(100, q=2, T=T, cs = cs)
                cold.run_simulation(M, M_sampling, method=method)
                ax.plot(cold.E[:cold.i:(cold.i//1000)]/N, label= str(methodsstring[i]) + ', cs ', linewidth=2)
                
                print(str(method), 'hot start final total energy: ', hot.e, 'cold start ', cs, ': ', cold.e) 
            
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Energy $E$ per spin')
            ax.legend()
            plt.savefig(pathname_plots/f"Hot-cold-starts_{T}_{q}.pdf", bbox_inches='tight') 
            tikzplotlib_fix_ncols(plt.gcf()) # workaround for bug in tikzplotlib
            tikzplotlib.save(pathname_plots/f"Hot-cold-starts_{T}_{q}.pgf")
            plt.show()

    # Check if the energy calculations coincide
    if False:
        print('Testing energy calculation.')
        model = Potts()
        model.run_simulation()
        model.test_energies()
            
    # Create a time series of the temperature with Bolzmann
    if False:
        Ts = [1, 0.696, 0.697, 0.698]
        M_tots = np.array([1E5, 1E6, 4E6, 1E7, 1E8])
        Ms = M_tots.copy()
        Ms[1:] -= M_tots[:-1]

        methods = [MC_step_fast]
        n_runs = 4
        for method in methods:
            model = Potts(300, q=10, T=1E2)
            model.run_simulation(-5000, 0, method=method)
            t_0 = model.i
            for i, M in enumerate(Ms):
                M_tot = int(M_tots[i])
                # run the simulation
                pf = time.perf_counter()
                model.run_simulation(M, method=method)
                print(f'The simulation with method {method.__name__} took {time.perf_counter()-pf} seconds.')
                model.write_E(pathname/f'Energies_maxwell_distribution_{method.__name__}_M{M_tot}.csv', t_0=t_0)
                # and plot the results
                E = np.loadtxt(pathname/f'Energies_maxwell_distribution_{method.__name__}_M{M_tot}.csv', delimiter=',')
                plot_energies_distr(E, filename=pathname_plots/f'Energies_maxwell_distribution_{method.__name__}_{i}', show_plt=True)
            E = np.loadtxt(pathname/f'Energies_maxwell_distribution_{method.__name__}_M{int(M_tots[-1])}.csv', delimiter=',')
            plot_energies_t0(E, filename=pathname/f'Energies_evolution_{method.__name__}_M{int(M_tots[-1])}')

    # TODO: This should work but doesn't
    # plot_energies(pathname/f'Energies_maxwell_distribution_MC_step_fast_M10000000.csv')

    # Create distributions without looping through an array but for both algorithms: 
        # Create a time series of the temperature with Bolzmann
    if False:
        M = -5000
        M_sampling = int(1E6)
        methods = [MC_step_fast, Gibbs_step]
        #n_runs = 4
        for method in methods:
            if False:
                # run the simulation
                model = Potts(300, q=10, T=100)
                filename= pathname/f'Energies_maxwell_distribution'
                model.run_simulation(M, M_sampling, method=method)
                model.write_E(filename)
            if False:
                # and plot it
                E = np.loadtxt(filename, delimiter=',')
                t_0 = int(len(E)-M_sampling)
                plot_energies_distr(E[t_0:], filename= pathname_plots/f'Energies_maxwell_distribution_{method.__name__}_{model.T}_{model.q}_{model.L}', show_plt=True)
                # plot_energies_t0(E, t_0)

    if False:
        # Show nice animations for high, medium and low temperatures
        Ts = [1E5, 1, 1E-1]
        filenames = [pathname_plots/f'{state}_temp_state' for state in ['High', 'Medium', 'Low']]
        for i in range(3):
            model = Potts(20, Ts[i], 5)
            model.run_simulation(10000, show_state=range(0,10000,200), save_state=[10000], filename=filenames[i])


    # Define the parameters for the experiments
    qs = [2,10]# range(2,10,3)
    Ts = np.linspace(0.1,2,40)
    M = -int(1E6)
    M_sampling = int(1E7)
    L = 5

    means = pd.DataFrame(columns=Ts, index=qs)
    stddev = pd.DataFrame(columns=Ts, index=qs)
    t_0s = pd.DataFrame(columns=Ts, index=qs) # time it takes to reach equilibrium

    if True:
        # Run the simulation for various T and q
        for q in qs:
            for T in Ts:
                print(f'running model for L={L}, T={T}, q={q}')
                # pf = time.perf_counter()
                model = Potts(L, T, q, cs = False)
                # print(f'setup {time.perf_counter()-pf}.')
                # pf = time.perf_counter()
                model.run_simulation(M, M_sampling, method=MC_step_fast)
                # print(f'running {time.perf_counter()-pf}.')
                means.loc[q][T], stddev.loc[q][T], t_0s.loc[q][T] = model.get_stats(M_sampling)
        means.to_pickle(pathname/f'means_L{L}.pkl')
        stddev.to_pickle(pathname/f'stddev_L{L}.pkl')
        t_0s.to_pickle(pathname/f't0s_L{L}.pkl')
    
    if True:

        plt.style.use(pathname_gen/'rc.mplstyle')
        # plot standard deviation and means
        means = pd.read_pickle(pathname/f'means_L{L}.pkl')
        stddev = pd.read_pickle(pathname/f'stddev_L{L}.pkl')
        fig, ax = plt.subplots()
        for q in qs:
            # plot the values in dependence of the temperature
            ax.errorbar(Ts, means.loc[q], yerr=stddev.loc[q], label=f'{q}')
        ax.legend(title='Parameter $q$', labels=qs)
        ax.set_xlabel('Temperature $T$')
        ax.set_ylabel('Energy $E$ per spin')
        ax.axvline(1.13, color='cornflowerblue', alpha = 0.75)
        ax.axvline(0.701, color='orange', alpha = 0.75)
        ax.text(1.15, -1.1, r"$T_c^{q = 2} = 1.13$")
        ax.text(0.71, -1.5, r"$T_c^{q = 10} = 0.701$")
        tikzplotlib_fix_ncols(plt.gcf()) # workaround for bug in tikzplotlib
        tikzplotlib.save(pathname_plots/f'energies_T_q_L{L}.pgf')
        ax.figure.savefig(pathname_plots/f'energies_T_q_L{L}.png', bbox_inches='tight')
        #plt.savefig(Path(pathname_plots/'MC')/f"Energy_Temperature_{L}_{M}_{M_sampling}.pdf" , bbox_inches='tight') # for Anna
        plt.show()

        # plot t_0s 
        t_0s = pd.read_pickle(pathname/f't0s_L{L}.pkl')

        fig, ax = plt.subplots()
        for q in qs:
            ax.plot(t_0s.loc[q], label=f'{q}')
        ax.legend(title='Parameter $q$', labels=qs)
        ax.set_xlabel('Temperature $T$')
        ax.set_ylabel('Time $t_0$')
        ax.axvline(1.13, color='orange', alpha = 0.75)
        ax.axvline(0.701, color='cornflowerblue', alpha = 0.75)
        #ax.text(1.15, 3E5, r"$T_c^{q = 2} = 1.13$")    #y - high possibly has to be addapted
        #ax.text(0.71, 3E5,  r"$T_c^{q = 10} = 0.701$")
        tikzplotlib_fix_ncols(plt.gcf()) # workaround for bug in tikzplotlib
        tikzplotlib.save(pathname_plots/f't0_T_q_L{L}.pgf')
        ax.figure.savefig(pathname_plots/f't0_T_q_L{L}.png',  bbox_inches='tight')
        #plt.savefig(Path(pathname_plots/'MC')/f"Time_Temperature_{L}_{M}_{M_sampling}_{q}_.pdf", bbox_inches='tight') # for ANna
        plt.show()
