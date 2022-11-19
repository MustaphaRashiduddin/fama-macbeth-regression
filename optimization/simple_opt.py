import prof_excel as pe
converged = False
import numpy as np
x_star = pe.x_star
theta = np.array([-1.0, -0.5556, -0.3333, 0.1111, -0.1111, -0.3333, 1.0, -1.0, -1.0, 0.3333])
epsilon = 0.1
def distance(x_star, w):
    return np.sqrt((x_star[0] - w[0])**2 +(x_star[1] - w[1])**2 +(x_star[2] - w[2])**2 +(x_star[3] - w[3])**2 +(x_star[4] - w[4])**2 +(x_star[5] - w[5])**2 +(x_star[6] - w[6])**2 +(x_star[7] - w[7])**2 +(x_star[8] - w[8])**2 +(x_star[9] - w[9])**2)
w = np.ndarray.flatten(pe.w)
theta = np.array([[-0.70122495], [0.41626324], [0.38880032], [-0.17164146], [0.49286745], [0.07179282], [0.09966174], [0.02637231], [0.88909471], [0.56627951]])
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = theta
space = np.linspace(-1,1,300)
tmp_d = distance(x_star(theta),w)
print(tmp_d)
p = 35
def optimize(tmp_d):
    for x6 in space:
        for x2 in space:
            for x3 in space:
                # for x4 in space:
                    # for x5 in space:
                        # for x6 in space:
                            # for x7 in space:
                                # for x8 in space:
                                    # for x9 in space:
                                        # for x10 in space:
                                            theta[0] = x1
                                            theta[1] = x2
                                            theta[2] = x3
                                            theta[3] = x4
                                            theta[4] = x5
                                            theta[5] = x6
                                            theta[6] = x7
                                            theta[7] = x8
                                            theta[8] = x9
                                            theta[9] = x10
                                            d = distance(x_star(theta), w)
                                            if (d < tmp_d):
                                                tmp_d = d
                                                print(d)
                                                print(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
                                                # print(round(x1, p), round(x2, p), round(x3, p), round(x4, p), round(x5, p), round(x6, p), round(x7, p), round(x8,p), round(x9,p), round(x10,p))
                                            # if (d <= epsilon):
                                                # quit()
optimize(tmp_d)
