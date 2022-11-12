import prof_excel as pe
from pymoo.core.problem import Problem
import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.moo.unsga3 import UNSGA3

x_star = pe.x_star
w = pe.w
start = np.divide(np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]),1.)
end = np.divide(np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]),1.)
theta=pe.theta
w = pe.w

class mproblem(Problem):
    def _evaluate(self, thetas, out, *args, **kwargs):
        _, _ = args, kwargs
        res_f = []
        res_g = []
        for theta in thetas:
            res_f.append(x_star(theta))
            res_g.append(w[9]-x_star(theta)[9])
        out["F"] = np.array(res_f)
        out["H"] = np.array(res_g)

stop_criteria = ('n_gen', 150)
problem = mproblem(n_var=10,n_obj=10,n_eq_constr=1,xl=start,xu=end)
ref_dirs=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
algorithm = UNSGA3(ref_dirs, pop_size=100)
results = minimize(problem=problem, 
                   algorithm=algorithm, 
                   termination=stop_criteria,
                   save_history=True,
                   seed=1,
                   verbose=True
                   )
print("results.X")
print(results.X)
print("----------")
print("results.F")
print(results.F)
print("----------")
print("w")
print(w)
