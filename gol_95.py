
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
    dM_dt = v_s * K_I^n/(K_I^n + P_N^n)  -  v_m * M / (K_m + M)
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

    :param params:
    :return:
    '''

    return