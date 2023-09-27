
import scipy as sc
import numpy as np
import matplotlib.pyplot as pl
import tikzplotlib
from pgf_plot_fix import tikzplotlib_fix_ncols

import matplotlib.ticker as ticker

# TODO:
# - implement backgrounds on graphs
# - implement the other graphs

# the following parameters were taken from Table 1

params_Mef = {
    'N_over':0,
    'O_over':0,
    'T_over':0.05,
    'LIF':0,
    'p_N':1,
    'p_O':1,
    'p_T':1,
    'K_d':0.1,
    'K_O':0.3,
    'K_NT':0.2,
    'n':2
}

# initialise experiments
params_default = params_Mef.copy()
params_default['LIF'] = 0.06
over_expressions = [0.1, 0.13, 0.2, 0.3]
expressions = ['N','O','T']

Experiment_labels = [f'{expr}_{over_expression}' 
                        for expr in expressions for over_expression in over_expressions]
LIF_expression = {f'{exp_label}':False for exp_label in Experiment_labels}

params = {exp_label:params_Mef.copy() 
          for exp_label in Experiment_labels}

# initialise values for experiments
for expr in expressions:
    for over_expression in over_expressions:
        exp_label = f'{expr}_{over_expression}'
        if expr != 'N' and over_expression >= 0.13:
            LIF_expression[exp_label] = True
            params[f'{expr}_{over_expression}']['LIF'] = 0.6
        params[exp_label][f'{expr}_over'] = over_expression

# experiment series
param_lists = {f'{exp_label}':[params_Mef, params[exp_label], params_default if LIF_expression[exp_label] else params_Mef]
                for exp_label in Experiment_labels}

# To make the plots look nice
NOT_labels = {'N':'Nanog', 'O':'Oct4', 'T':'Tet1'}
colors = {'N':'blue', 'O':'green', 'T':'orange'}


def rhs(y, params=params_Mef):
    # y is of the form (N, O, T)
    # params id of the form of default_params
    N = np.take(y,0)
    O = np.take(y,1)
    T = np.take(y,2)

    O_KO = O/params['K_O']
    NT = (params['K_d']+N+T)/2-np.sqrt(((params['K_d']+N+T)/2)**2-N*T)
    NT_KNT2 = (NT/params['K_NT'])**params['n']
    return np.array([params['N_over']+params['LIF']+params['p_N']*O_KO/(1+O_KO)-N,
                  params['O_over']+params['LIF']+params['p_O']*NT_KNT2/(1+NT_KNT2)*O_KO/(1+O_KO)-O,
                  params['T_over']+params['p_T']*NT_KNT2/(1+NT_KNT2)*O_KO/(1+O_KO)-T])

running_time = 30

def run_experiment_series(param_list):
    t = np.empty(0)
    y = np.zeros(shape=(3,0))
    for param in param_list:
        y0 = np.zeros(3) if not y.shape[1] else y[:,-1]
        t0 = 0 if not len(t) else t[-1]
        soln = sc.integrate.solve_ivp(lambda t,y: rhs(y, param), (t0,t0+running_time), y0, vectorised=True, max_step=0.1)
        t = np.append(t, soln.t)
        y = np.append(y, soln.y, axis=1)
    return t, y



if __name__ == '__main__':

    for j, exp_label in enumerate(Experiment_labels):
        fig, ax = pl.subplots()
        ax.set_xlim((0,3*running_time))
        ax.set_ylim(0,1)
        ax.set_yticks([0,1])
        ax.set_xticks([i*running_time for i in range(4)], labels=['' for i in range(4)])
        t, y = run_experiment_series(param_lists[exp_label])
        for i, expr in enumerate(expressions):
            pl.plot(t, y[i,:], label=NOT_labels[f'{expr}'], color=colors[f'{expr}'])

        pl.legend()
        tikzplotlib_fix_ncols(fig)

        ax.set_xlabel('Time')
        ax.set_ylabel('Expression level')
        ax.set_xticks([(i+0.5)*running_time for i in range(3)], labels=[f'${exp_label[0]}_{{over}}=0$', f'${exp_label[0]}_{{over}}={exp_label[2:]}$', f'${exp_label[0]}_{{over}}=0$'], minor=True)
        # add background for experiment
        ax.axvspan(running_time, 2*running_time, alpha=0.2, color=colors[exp_label[0]], label=f'{NOT_labels[exp_label[0]]} overexpressed')
        if LIF_expression[exp_label]:
            ax.axvspan(running_time, 3*running_time, alpha=0.2, color='gray', label='LIF active')
            ax.set_xticks([(i+0.5)*running_time for i in range(3)], labels=[f'{exp_label[0]}=0, LIF=0', f'{exp_label[0]}={exp_label[2:]}, LIF=0.06', f'{exp_label[0]}=0, LIF=0.06'], minor=True)
        

        tikzplotlib.save(f'Plots/{exp_label}.pgf')
    
    pl.show()
