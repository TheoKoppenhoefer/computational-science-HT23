
import scipy as sc
import numpy as np
import matplotlib.pyplot as pl

# the following parameters were taken from Table 1

default_params = {
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

params_N = default_params.copy()
params_N['N_over'] = 0.3

params_O = default_params.copy()
params_O['O_over'] = 0.3

params_T = default_params.copy()
params_T['T_over'] = 0.3

def rhs(y, params=default_params):
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


def run_experiment_series(param_list):
    t = np.empty(0)
    y = np.zeros(shape=(3,0))
    for param in param_list:
        y0 = np.zeros(3) if not y.shape[1] else y[:,-1]
        t0 = 0 if not len(t) else t[-1]
        soln = sc.integrate.solve_ivp(lambda t,y: rhs(y, param), (t0,t0+30), y0, vectorised=True, max_step=0.1)
        t = np.append(t, soln.t)
        y = np.append(y, soln.y, axis=1)
    return t, y



if __name__ == '__main__':
    NOT_labels = ['Nanog', 'Oct4', 'Tet1']

    # run simulation for over expression of O
    # experiment series
    param_lists = [[default_params, params_O, default_params],
                   [default_params, params_N, default_params],
                   [default_params, params_T, default_params]]
    for param_list in param_lists:
        t, y = run_experiment_series(param_list)
        for i, label in enumerate(NOT_labels):
            pl.plot(t, y[i,:], label=label)
        pl.legend()
        pl.show()