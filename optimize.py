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
w = pe.w
start = np.multiply(np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]),1.)
end = np.multiply(np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]),1.)
theta=pe.theta
w = pe.w

class mproblem(Problem):
    def _calc_pareto_front(self,*args, **kwargs):
        _, _ = args, kwargs
        return w

    def _evaluate(self, thetas, out, *args, **kwargs):
        _, _ = args, kwargs
        res_f = []
        # res_h = []
        # res_g = []
        for theta in thetas:
            res_f.append(x_star(theta))
            # res_h.append(w-x_star(theta))
            # res_g.append(-x_star(theta))
            # x_star(theta) >= 0
            # res_g.append(x_star(theta)-1)
            # x_star(theta) <= 1
        out["F"] = np.array(res_f)
        # out["H"] = np.array(res_h)
        # out["G"] = np.array(res_g)

stop_criteria = ('n_gen', 200)
problem = mproblem(n_var=10,n_obj=10,n_eq_constr=0,n_ieq_constr=0,xl=start,xu=end)

ref_dirs=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# ref_dirs = get_reference_directions("das-dennis", 10, n_partitions=3)

# algorithm = RVEA(ref_dirs)
# algorithm = SMSEMOA()
algorithm = UNSGA3(ref_dirs)
results = minimize(problem=problem, 
                   algorithm=algorithm, 
                   termination=stop_criteria,
                   # save_history=True,
                   seed=1,
                   pf=problem.pareto_front(),
                   verbose=False
                   )
print("results.X")
print(results.X)
print("----------")
print("results.F")
print(results.F)
print("----------")
print("w")
print(w)
