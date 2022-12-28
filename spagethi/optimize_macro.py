import prof_excel as pe
from pymoo.core.problem import Problem
import numpy as np
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.unsga3 import UNSGA3

x_star = pe.x_star
start = np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05])
end = np.array([.15,.15,.15,.15,.15,.15,.15,.15,.15,.15])
cwc = pe.cwc

class mproblem(Problem):
    def _evaluate(self, thetas, out, *args, **kwargs):
        _, _ = args, kwargs
        res_f = []
        res_h = []
        res_g = []
        for theta in thetas:
            thetaT = np.array([theta])
            thetaT = thetaT.T
            cwc_memoized = cwc(thetaT)
            res_f.append(cwc_memoized[10])
            res_h.append(1-np.sum(cwc_memoized[0:10]))
            res_g.append(-cwc_memoized[0:10])
            res_g.append(cwc_memoized[0:10]-1)
        out["F"] = np.array(res_f)
        out["H"] = np.array(res_h)
        out["G"] = np.array(res_g)

stop_criteria = ('n_gen', 270)
problem = mproblem(n_var=10,n_obj=1,n_eq_constr=1,n_ieq_constr=20,xl=start,xu=end)

ref_dirs=np.array([[0.]])
ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=150)
algorithm = UNSGA3(ref_dirs, pop_size=15000)
results = minimize(problem=problem, 
                   algorithm=algorithm, 
                   termination=stop_criteria,
                   # save_history=True,
                   # seed=1,
                   return_least_infeasible=True,
                   verbose=True
                   )
print("results.X")
print(results.X)
print("----------")
print("results.F")
print(results.F)
print("----------")
resultsXT = np.array([results.X])
resultsXT = resultsXT.T
print("x_star(resultsXT)")
print(x_star(resultsXT))
# print("{:.15f}".format(x_star(resultsXT)))
print("----------")
print("np.sum(x_star(resultsXT))")
print(np.sum(x_star(resultsXT)))
