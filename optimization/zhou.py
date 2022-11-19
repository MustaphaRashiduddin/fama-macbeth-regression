# populate w.csv with optimum_weight function

from numpy import e

# lmd = 1-r.csv
# M, R are matrices with same shape
def optimum_weight(pi, M, tau, R, eta, lmd, T, theta, t):
    return pi*M*e**(T*tau - t*tau)/R + eta*(M*e**(T*tau - t*tau)/R
                                            - 1 + e**(-0.5*T*eta*theta**T +
                                                      0.5*eta*t*theta**T)/(R*lmd))
