# this example is checked to be true beforehand ie how to simplify integrals with limits

from sympy import symbols
from sympy import Integral
from sympy import integrate
from sympy import pprint

x = symbols('x')
eqn_integral = Integral(x**3, (x, 2, 4))
pprint(eqn_integral)
eqn_integrated = integrate(x**3, (x, 2, 4))
pprint(eqn_integrated)
