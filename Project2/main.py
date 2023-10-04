
import scipy as sc
import numpy as np
import matplotlib.pyplot as pl
import tikzplotlib
from pgf_plot_fix import tikzplotlib_fix_ncols

import matplotlib.ticker as ticker

# the following parameters were taken from Table 1 in Olariu, 2016
params_MEF = {
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
params_LIF = params_MEF.copy()
params_LIF['LIF'] = 0.06
over_expressions = [0.1, 0.13, 0.2, 0.3] # 
over_expressions_repressed = [0.01,0.1,0.8]
expressions = ['N','O','T']

Experiment_labels = [f'{expr}_{over_expression}' 
                        for expr in expressions for over_expression in over_expressions]
# Add experiments to the list
Experiment_labels += ['NO_0.1']
Experiment_labels += [f'repressed_{over_expression}' for over_expression in over_expressions_repressed]
LIF_expression = {f'{exp_label}':False for exp_label in Experiment_labels}
repression = {f'{exp_label}':False for exp_label in Experiment_labels}

params = {exp_label:params_MEF.copy() 
          for exp_label in Experiment_labels}

# initialise values for experiments
for expr in expressions:
    for over_expression in over_expressions:
        exp_label = f'{expr}_{over_expression}'
        if expr != 'N' and over_expression >= 0.13:
            LIF_expression[exp_label] = True
            params[f'{expr}_{over_expression}']['LIF'] = 0.06
        params[exp_label][f'{expr}_over'] = over_expression

# This adds the experiment with repression (i.e. the additional exercise)
for over_expression in over_expressions_repressed:
    exp_label = f'repressed_{over_expression}'
    params[exp_label][f'O_over'] = over_expression
    params[exp_label][f'p_N'] = 5
    repression[exp_label] = True

# This adds plot b)
params['NO_0.1']['O_over'] = 0.1
params['NO_0.1']['N_over'] = 0.2
params['NO_0.1']['LIF'] = 0.06
LIF_expression['NO_0.1'] = True

# experiment series
param_lists = {f'{exp_label}':[params_MEF, params[exp_label], params_LIF if LIF_expression[exp_label] else params_MEF]
                for exp_label in Experiment_labels}

# To make the plots look nice
NOT_labels = {'N':'Nanog', 'O':'Oct4', 'T':'Tet1'}
colors = {'N':'blue', 'O':'green', 'T':'orange'}

# Add the LIF withdrawal experiment (plot c)
Experiment_labels += ['LIF_withdrawal']
repression['LIF_withdrawal'] = False
param_lists['LIF_withdrawal'] = [params_LIF, params_MEF]

Experiment_labels += ['LIF_withdrawal_0']
repression['LIF_withdrawal_0'] = False
params_zero = params_MEF.copy()
params_zero['T_over'] = 0
param_lists['LIF_withdrawal_0'] = [params_LIF, params_zero]

def rhs(y, params=params_MEF, repression=False):
    # y is of the form (N, O, T)
    # params id of the form of default_params
    N = np.take(y,0)
    O = np.take(y,1)
    T = np.take(y,2)

    O_KO = O/params['K_O']
    NT = (params['K_d']+N+T)/2-np.sqrt(((params['K_d']+N+T)/2)**2-N*T)
    NT_KNT2 = (NT/params['K_NT'])**params['n']
    if not repression: return np.array([params['N_over']+params['LIF']+params['p_N']*O_KO/(1+O_KO)-N,
                  params['O_over']+params['LIF']+params['p_O']*NT_KNT2/(1+NT_KNT2)*O_KO/(1+O_KO)-O,
                  params['T_over']+params['p_T']*NT_KNT2/(1+NT_KNT2)*O_KO/(1+O_KO)-T])
    U = np.take(y,3)
    V = np.take(y,4)
    U_KU = U/0.2
    V_KV = V/0.2
    return np.array([params['N_over']+params['LIF']+params['p_N']*O_KO/(1+O_KO)/(1+U_KU)/(1+V_KV)-N,
                  params['O_over']+params['LIF']+params['p_O']*NT_KNT2/(1+NT_KNT2)*O_KO/(1+O_KO)-O,
                  params['T_over']+params['p_T']*NT_KNT2/(1+NT_KNT2)*O_KO/(1+O_KO)-T,
                  O_KO/(1+O_KO)-U,
                  O_KO/(1+O_KO)-V])

running_time = 40

def run_experiment_series(param_list, y = np.zeros(shape=(3,1)), repression=False):
    t = np.empty(0)
    for param in param_list:
        y0 = y[:,-1]
        t0 = 0 if not len(t) else t[-1]
        soln = sc.integrate.solve_ivp(lambda t,y: rhs(y, param, repression), (t0,t0+running_time), y0, vectorised=True, max_step=0.1)
        t = np.append(t, soln.t)
        y = np.append(y, soln.y, axis=1)
    return t, y[:,1:]



if __name__ == '__main__':
    show_plots=False

    for j, exp_label in enumerate(Experiment_labels):
        run_length = len(param_lists[exp_label])
        fig, ax = pl.subplots()
        ax.set_xlim((0,run_length*running_time))
        ax.set_ylim(0,1.05)
        ax.set_yticks([0,1])
        
        ax.set_xticks([i*running_time for i in range(run_length+1)], labels=['' for i in range(run_length+1)], minor=True)
        # set x tick labels
        xlabels = [''.join([f'${k}={params[k]}$' if params[k]!=params_MEF[k] else '' for k in params_MEF]) for params in param_lists[exp_label]] 
        xlabels = [label.replace('_over',r'_\text{over}').replace('LIF',r'\text{LIF}').replace('$$','$, $') for label in xlabels]
        if show_plots: xlabels = [label.replace(r'\text','') for label in xlabels]

        ax.set_xticks([(i+0.5)*running_time for i in range(run_length)], labels=xlabels)
        y0 = 0.7*np.ones((5 if repression[exp_label] else 3,1)) if exp_label[:14] == 'LIF_withdrawal' else np.zeros((5 if repression[exp_label] else 3,1))
        t, y = run_experiment_series(param_lists[exp_label], y0, repression[exp_label])

        for i, expr in enumerate(expressions):
            ax.plot(t[::(1000//len(t)+1)], y[i,::(1000//len(t)+1)], label=NOT_labels[f'{expr}'], color=colors[f'{expr}'], linewidth=2)


        ax.set_xlabel('Time')
        ax.set_ylabel('Expression level')
        # add background for experiment
        for k, param in enumerate(param_lists[exp_label]):
            if params_MEF['LIF'] != param['LIF']: ax.axvspan(k*running_time, (k+1)*running_time, alpha=0.2, color='gray', label='LIF active')  
        
            for expr in expressions:
                if params_MEF[f'{expr}_over'] < param[f'{expr}_over']:
                    ax.axvspan(k*running_time, (k+1)*running_time, alpha=0.2, color=colors[expr], label=f'{NOT_labels[expr]} overexpressed')

        # Add a legend
        pl.legend()
        tikzplotlib_fix_ncols(fig)

        if not show_plots: tikzplotlib.save(f'Plots/{exp_label}.pgf')

    if show_plots: pl.show()
        
    