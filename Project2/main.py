
import scipy as sc
import numpy as np
import matplotlib.pyplot as pl
import tikzplotlib
from pgf_plot_fix import tikzplotlib_fix_ncols

import matplotlib.ticker as ticker

# TODO:
# - add background to graphs to legend
# - implement the other graphs
# - fix issue with >= 1

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
# Add experiments to the list
Experiment_labels += ['NO_0.1']
LIF_expression = {f'{exp_label}':False for exp_label in Experiment_labels}

params = {exp_label:params_Mef.copy() 
          for exp_label in Experiment_labels}

# initialise values for experiments
for expr in expressions:
    for over_expression in over_expressions:
        exp_label = f'{expr}_{over_expression}'
        if expr != 'N' and over_expression >= 0.13:
            LIF_expression[exp_label] = True
            params[f'{expr}_{over_expression}']['LIF'] = 0.06
        params[exp_label][f'{expr}_over'] = over_expression

# This adds plot b)
params['NO_0.1']['O_over'] = 0.1
params['NO_0.1']['N_over'] = 0.3
LIF_expression['NO_0.1'] = True

# experiment series
param_lists = {f'{exp_label}':[params_Mef, params[exp_label], params_default if LIF_expression[exp_label] else params_Mef]
                for exp_label in Experiment_labels}

# To make the plots look nice
NOT_labels = {'N':'Nanog', 'O':'Oct4', 'T':'Tet1'}
colors = {'N':'blue', 'O':'green', 'T':'orange'}

# Add the LIF withdrawal experiment (plot c)
Experiment_labels += ['LIF_withdrawal']
param_lists['LIF_withdrawal'] = [params_default, params_Mef]

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

running_time = 40

def run_experiment_series(param_list, y = np.zeros(shape=(3,1))):
    t = np.empty(0)
    for param in param_list:
        y0 = y[:,-1]
        t0 = 0 if not len(t) else t[-1]
        soln = sc.integrate.solve_ivp(lambda t,y: rhs(y, param), (t0,t0+running_time), y0, vectorised=True, max_step=0.1)
        t = np.append(t, soln.t)
        y = np.append(y, soln.y, axis=1)
    return t, y[:,1:]



if __name__ == '__main__':

    for j, exp_label in enumerate(Experiment_labels):
        run_length = len(param_lists[exp_label])
        fig, ax = pl.subplots()
        ax.set_xlim((0,run_length*running_time))
        ax.set_ylim(0,1.05)
        # ax.axes.set_aspect(3)
        ax.set_yticks([0,1])
        ax.set_xticks([i*running_time for i in range(run_length+1)], labels=['' for i in range(run_length+1)], minor=True)
        # set x tick labels
        if exp_label == 'LIF_withdrawal':
            xlabels = ['$LIF=0.06$','$LIF=0$']
        else:
            xlabels = [f'${exp_label[0]}_\\text{{over}}\!=0$, $\\text{{LIF}}=0$', f'${exp_label[0]}_\\text{{over}}\!={exp_label[2:]}$, $\\text{{LIF}}=0.06$', f'${exp_label[0]}_\\text{{over}}\!=0$, $\\text{{LIF}}=0.06$'] \
                            if LIF_expression[exp_label] else [f'${exp_label[0]}_\\text{{over}}\!=0$', f'${exp_label[0]}_\\text{{over}}\!={exp_label[2:]}$', f'${exp_label[0]}_\\text{{over}}\!=0$']


        ax.set_xticks([(i+0.5)*running_time for i in range(run_length)], labels=xlabels)
        y0 = 0.7*np.ones((3,1)) if exp_label == 'LIF_withdrawal' else np.zeros((3,1))
        t, y = run_experiment_series(param_lists[exp_label], y0)

        for i, expr in enumerate(expressions):
            pl.plot(t[::(1000//len(t)+1)], y[i,::(1000//len(t)+1)], label=NOT_labels[f'{expr}'], color=colors[f'{expr}'], linewidth=2)


        ax.set_xlabel('Time')
        ax.set_ylabel('Expression level')
        # add background for experiment
        if exp_label == 'LIF_withdrawal':
            ax.axvspan(0, running_time, alpha=0.2, color='gray', label='LIF active')  
        else:
            ax.axvspan(running_time, 2*running_time, alpha=0.2, color=colors[exp_label[0]], label=f'{NOT_labels[exp_label[0]]} overexpressed')
            if LIF_expression[exp_label]:
                ax.axvspan(running_time, 3*running_time, alpha=0.2, color='gray', label='LIF active')  

        # Add a legend
        pl.legend()
        tikzplotlib_fix_ncols(fig)

        tikzplotlib.save(f'Plots/{exp_label}.pgf')
        

    
    #pl.show()
