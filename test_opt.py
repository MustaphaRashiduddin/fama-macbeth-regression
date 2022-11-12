from pymoo.core.problem import Problem
import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.misc import parameter_less

start = np.array([-100.,-100.])
end = np.array([100.,100.])

# results.X = [[49.51007474 62.90397173]]
ans = np.array([[128.27697388, 31.93264014]])

class mproblem(Problem):

    def _evaluate(self, x, out, *args, **kwargs):
        _, _ = args, kwargs
        f1 = (x[:,0]-5)**2
        f2 = (x[:,1]-6)**2
        out["F"] = np.array([f1,f2])
        # print(out["F"])
        # print("---")
        # print(np.array([(ans[0])]).T)
        # print("----")
        # print(out["F"]-np.array([(ans[0])]).T)
        # quit()
        g1 = x[:,0]**2-4
        g2 = np.exp(-x[:,0])-x[:,1]
        g3 = -x[:,0]
        g4 = -x[:,1]
        out["G"] = np.array([g1,g2,g3,g4])
        # out["H"] = out["F"]-np.array([(ans[0])]).T

stop_criteria = ('n_gen', 150)
problem = mproblem(n_var=2,n_obj=2,n_eq_constr=0, n_ieq_constr=4,xl=start,xu=end)
ref_dirs=np.array([[0., 0.]])
algorithm = UNSGA3(ref_dirs, pop_size=100)
# algorithm = NSGA2(pop_size=100)
results = minimize(problem=problem, 
                   algorithm=algorithm, 
                   termination=stop_criteria,
                   save_history=True,
                   seed=1,
                   verbose=True
                   )
# print(problem.pareto_front(theta))
print("results.X")
print(results.X)
print("----------")
print("results.F")
print(results.F)
