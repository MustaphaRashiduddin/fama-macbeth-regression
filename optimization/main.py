import numpy as np
# lft_lft_exp = (1/R * e(f_lft_integral())
# lft_exp = (lft_lft_exp + lft_mid_exp - 1) * eta

# 2014m12
# T = 12
# 2015m12
# # T = 2014m12

T = 2014m10
print(1/R * e(summation_ft_integral()))

dates = np.array(['2014m10', '2014m11', '2014m12', '2015m1', '2015m2', '2015m3', '2015m4', '2015m5', '2015m6', '2015m7', '2015m8', '2015m9'])

def f_lft_integral():
    return 1/R * e(summation_ft_integral())

def f_summation_lft_integral():
    t = dates[0]
    i = len(dates)-1
    T = dates[i]
    res_summation_lft_integral = np.zeros((1, 10))
    while T != t:
        T = dates[i]
        i = i-1
        res_summation_lft_integral += f_lft_integral()
    res_summation_lft_integral += f_lft_integral()
