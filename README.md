# fama-macbeth-regression

run the optimize.py file or the macro variant of it in the spaghetti folder, the prof_excel.py calculates the distance of w from x* and provides
all decision variables of x* to optimizer such that constraints can be enforced, i.e. all variables must be between 0 < x*[i] < 1, 
and the summation of x*[i] = 1

you need to set T between 1 and 12, the starting date and the fund manage in the code. no user friendly handles provided.
