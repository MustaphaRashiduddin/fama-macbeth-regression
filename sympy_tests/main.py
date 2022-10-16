# export to professor's code

from numpy import pi
from numpy import e

def optimum_weight(M, tau, R, eta, lmd, T, theta, t):
    return pi*M*e*(T*tau - t*tau)/R + eta*(M*e*(T*tau - t*tau)/R - 1 - 
                                           e*(0.5*T*eta*theta**T - 0.5*eta*t*theta**T)/(R*lmd))
