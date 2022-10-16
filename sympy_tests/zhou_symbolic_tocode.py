# conversion of zhou's formula to python

from sympy import symbols
from sympy import pprint
from sympy import Integral
from sympy import integrate
from sympy import pi

theta, n, s, T, t, tau, lmd, R, e, M = symbols('theta, eta, s, T, t, tau, lambda, R, e, M')

def integration(exp, cnst, solve_integral):
    if solve_integral == True:
        return integrate(exp, cnst)
    else:
        return Integral(exp, cnst)

def make_integration_expr(coeff, expr, cnst, solve_integral):
    integral = integration(expr, cnst, solve_integral)
    return coeff * integral

# zhou's equation looking ambiguous don't know if integrals are powers or not so i made these two defs

def get_expr_1(solve_integral): # maybe integrals are powers
    integral1 = make_integration_expr(1, 1/2*theta**T*n, (s,t,T), solve_integral)
    integral2 = make_integration_expr(1, tau, (s,t,T), solve_integral)
    return ((e**-integral1)/(lmd*R)+(e**integral2)*M/R-1)*n+M/R*e**integral2*pi

def get_expr_2(solve_integral): # maybe not
    integral1 = make_integration_expr(1, 1/2*theta**T*n, (s,t,T), solve_integral)
    integral2 = make_integration_expr(M/R*e, tau, (s,t,T), solve_integral)
    expr1 = (e/(lmd*R)-integral1 + integral2 - 1) * n
    expr2 = integral2 * pi
    return expr1 + expr2

get_expr = get_expr_1 # change to get_expr_1 or get_expr_2
expr_unsimplified = get_expr(False)
expr_simplified = get_expr(True)

print("*ORIGINAL FORM*")
pprint(expr_unsimplified)
print()
print("*INTEGRALS SOLVED; (EXPRESSED AS PYTHON EXPRESSION IN NEXT STEP)*")
pprint(expr_simplified)
print()
print("*PLUG INTO FUNCTION AND IMPORT INTO PROFFESOR'S CODE*")
print(expr_simplified)
