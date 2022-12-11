# from _typeshed import ProfileFunction
# from pymoo.algorithms.moo.sms import SMSEMOA
import prof_excel as pe
from pymoo.core.problem import Problem
import numpy as np
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

################################################  ALREADY TRIED
from pymoo.algorithms.moo.unsga3 import UNSGA3
# from pymoo.algorithms.moo.rnsga3 import RNSGA3
# from pymoo.algorithms.moo.moead import 
# from pymoo.algorithms.moo.age2 import AGEMOEA2
# from pymoo.algorithms.moo.age2 import AGEMOEA2
# from pymoo.algorithms.moo.ctaea import CTAEA
# from pymoo.algorithms.soo.nonconvex.isres import ISRES
# from pymoo.algorithms.moo.rvea import RVEA
################################################

x_star = pe.x_star
cwc = pe.cwc
start = np.multiply(np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]),1.)
end = np.multiply(np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]),1.)

class mproblem(Problem):
    def _evaluate(self, thetas, out, *args, **kwargs):
        _, _ = args, kwargs
        res_f = []
        res_h = []
        res_g = []
        for theta in thetas:
            res_f.append(cwc(theta)[10])
            res_h.append(1-np.sum(cwc(theta)[0:10]))
            res_g.append(-cwc(theta)[0:10])
            res_g.append(cwc(theta)[0:10]-1)
        out["F"] = np.array(res_f)
        out["H"] = np.array(res_h)
        out["G"] = np.array(res_g)

stop_criteria = ('n_gen', 500)
problem = mproblem(n_var=10,n_obj=1,n_eq_constr=1,n_ieq_constr=20,xl=start,xu=end)

# ref_dirs=np.array([[0.]])
ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=100000)

# algorithm = RVEA(ref_dirs)
# algorithm = SMSEMOA()
algorithm = UNSGA3(ref_dirs, pop_size=1000)
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
print(x_star(results.X))
print(np.sum(x_star(results.X)))
