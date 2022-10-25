# conversion of zhou's formula to python

from sympy import symbols
# from sympy import Symbol
# from sympy.solvers import solve
from sympy import pprint
from sympy import Integral
from sympy import integrate
# from sympy import pi

theta_T, n, s, T, t, tau, lmd, R, e, M = symbols('theta_T, eta, s, T, t, tau, lambda, R, e, M')
theta, pi, y, b, w_b, K, w = symbols('theta, pi, gamma, beta  w_b, K, w')
# tau = M - 0.5*theta_T*n-theta_T*pi
# n = K * theta
# M = (y+theta)*w_b
# R = (y+theta)*w
lmd = 1

def integration(exp, cnst, solve_integral):
    if solve_integral == True:
        return integrate(exp, cnst)
    else:
        return Integral(exp, cnst)

def make_integration_expr(coeff, expr, cnst, solve_integral):
    integral = integration(expr, cnst, solve_integral)
    return coeff * integral

# zhou's equation looking ambiguous don't know if integrals are powers or not so i made these two defs

def get_expr(solve_integral): # maybe integrals are powers
    integral1 = make_integration_expr(1, 1/2*theta_T*n, (s,t,T), solve_integral)
    integral2 = make_integration_expr(1, tau, (s,t,T), solve_integral)
    return ((e**-integral1)/(lmd*R)+(e**integral2)*M/R-1)*n+M/R*e**integral2*pi

expr_unsimplified = get_expr(False)
expr_simplified = get_expr(True)
# pprint(expr_simplified)
pprint(expr_unsimplified)

# pprint(solve(expr_simplified, theta))
# pprint(solve(expr_unsimplified - w, theta_T))

# print("*ORIGINAL FORM*")
# pprint(expr_unsimplified)
# print()
# print("*INTEGRALS SOLVED; (EXPRESSED AS PYTHON EXPRESSION IN NEXT STEP)*")
# pprint(expr_simplified)
# print()
# print("*PLUG INTO FUNCTION AND IMPORT INTO PROFFESOR'S CODE*")
# print(expr_simplified)
