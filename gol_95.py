
'''
Project 3 Code
'''

import scipy
import numpy as np
import matplotlib.pyplot as plt


# model
def gol_95(t,y,params):
    '''

    '''

    # Unpacking Params(18)
    v_s = params[0]
    v_m = params[1]
    K_m = params[2]
    k_s = params[3]
    v_d = params[4]
    k_1 = params[5]
    k_2 = params[6]
    K_I = params[7]
    K_d = params[8]
    n = params[9]
    K_1 = params[10]
    K_2 = params[11]
    K_3 = params[12]
    K_4 = params[13]
    V_1 = params[14]
    V_2 = params[15]
    V_3 = params[16]
    V_4 = params[17]

    #Unpacking Y
    M = y[0]
    P_0 = y[1]
    P_1 = y[2]
    P_2 = y[3]
    P_N = y[4]

    #Calculating the 5 differential equations
    dM_dt = v_s * K_I**n/(K_I**n + P_N**n)  -  v_m * M / (K_m + M)
    dP_0_dt = k_s * M  -  V_1*P_0/(K_1+P_0)  +  V_2 * P_1/(K_2+P_1)
    dP_1_dt =  V_1*P_0/(K_1+P_0) - V_2 * P_1/(K_2+P_1) - V_3 * P_1/(K_3+P_1) + V_4*P_2/(K_4+P_2)
    dP_2_dt = V_3 * P_1/(K_3+P_1) - V_4*P_2/(K_4+P_2) - k_1*P_2 + k_2*P_N - v_d * P_2/(K_d+P_2)
    dP_N_dt = k_1*P_2 - k_2*P_N

    return (dM_dt,dP_0_dt,dP_1_dt,dP_2_dt,dP_N_dt)

def get_period(t, x):
    """ Approximate the period of a 1-D x, given the time-steps t.
        Returns a tuple with the period and the standard deviation of
the period over time.
        if the value of the standard deviation is not smaller than 0.1,
then
        it means the period estimate is dodgy and you shouldn't use it.
Instead,
        plot your simulation and figure out why it isn't periodic -
maybe it just
        hasn't reached the limit cycle yet."""
    idxs = scipy.signal.find_peaks(x)
    idxs = idxs[0]
    times = t[idxs]
    periods = np.diff(times)
    period = periods.mean()
    sdperiod = periods.std()

    return (period, sdperiod)



def gol95_cost( params ):
    '''
    :param params: the ndarray of parameters used to simulate the gol model. (For gol_95, this is 18 parameters)
    :return: the cost of the gol_95 model

    This function will:
        1. Run the simulation with params as the parameters for at least 10 days,
           so that it is likely to have reached the limit cycle.
        2. Re-run the simulation, beginning with the values from the final
           time step of the previous gol_95 simulation
        3. Compute the period and the cycle-to-cycle standard deviation of the period
           by calling get_period.
        4. Compute the cost with the formula cost = (((period - 23.6)/23.6)^2 + sdperiod/23.26)^0.5
    '''

    # Run the simulation with params as the parameters for at least 10 days,
    # so that it is likely to have reached the limit cycle. (Here each timstep is an hour)

    # Initial conditions
    M_0 = 1
    P_0_0 = 1
    P_1_0 = 1
    P_2_0 = 1
    P_N_0 = 1
    y0 = (M_0,P_0_0,P_1_0,P_2_0,P_N_0)


    # Time points
    days_to_run = 10
    t = np.linspace(0,24*days_to_run,24*days_to_run)

    # Run the simulation
    sol = scipy.integrate.solve_ivp(lambda t,y: gol_95(t,y,params),[0,24*days_to_run],y0,method='RK45',t_eval=t)

    ### TODO Question when is this used
    # sol2 = scipy.integrate.odeint(gol_95, y0, t, args=(params,))

    # Re-running the simulation, beginning with the values from the final timestep of the previous
    # gol_95 simulation
    y0 = sol.y[:,-1]
    sol = scipy.integrate.solve_ivp(lambda t, y: gol_95(t, y, params), [0, 24 * days_to_run], y0, method='RK45',
                                    t_eval=t)

    # Computing the period and the cycle-to-cycle standard deviation of the period




    print(2)




    return



def main():
    # Setting up the gol_95 parameters
    v_s = 0.76
    v_m = 0.65
    K_m = 0.5
    k_s = 0.38
    v_d = 0.95
    k_1 = 1.9
    k_2 = 1.3
    K_I = 1
    K_d = 0.2
    n = 4
    K_1 = 2
    K_2 = 2
    K_3 = 2
    K_4 = 2
    V_1 = 3.2
    V_2 = 1.58
    V_3 = 5
    V_4 = 2.5
    gol_95_params = (v_s, v_m, K_m, k_s, v_d, k_1, k_2, K_I, K_d, n,
                     K_1, K_2, K_3, K_4, V_1, V_2, V_3, V_4)

    gol95_cost(gol_95_params)


if __name__ == '__main__':
    main()