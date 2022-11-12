import prof_excel as pe
from pymoo.core.problem import Problem
import numpy as np
from pymoo.optimize import minimize
# from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
# from pymoo.algorithms.soo.nonconvex.sres import SRES

x_star = pe.x_star
w = pe.w
# theta = np.array([0.33333333333333326, -0.7777777777777778, -0.7777777777777778, -1.0, -0.33333333333333337, 0.7777777777777777, 1.0, -0.7777777777777778, 1.0, 0.11111111111111115])
start = np.divide(np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]),1.)
end = np.divide(np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]),1.)
theta=pe.theta
w = pe.w

class mproblem(Problem):

    def _calc_pareto_front(self):
        return w

    def _evaluate(self, thetas, out, *args, **kwargs):
        _, _ = args, kwargs
        res_f = []
        res_g = []
        for theta in thetas:
            res_f.append(x_star(theta))
            res_g.append(np.subtract(w,x_star(theta)))
        out["F"] = np.array(res_f)
        g = np.array(res_g)
        out["H"] = np.subtract(g,out["F"])

stop_criteria = ('n_gen', 1500)
problem = mproblem(n_var=10,n_obj=10,n_eq_constr=10,xl=start,xu=end)
ref_dirs=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
algorithm = UNSGA3(ref_dirs, pop_size=100)
# algorithm = NSGA2(pop_size=100)
results = minimize(problem=problem, 
                   algorithm=algorithm, 
                   termination=stop_criteria,
                   save_history=True,
                   seed=1,
                   pf=problem.pareto_front(),
                   verbose=True
                   )
# print(problem.pareto_front(theta))
print("results.X")
print(results.X)
print("----------")
print("results.F")
print(results.F)
print("----------")
print("w")
print(w)
