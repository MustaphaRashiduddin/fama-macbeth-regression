# populate w.csv with optimum_weight function
import numpy as np
# from scipy.integrate import quad

# def left_integrand(s, thetaT, k, theta):
    # return thetaT*k*theta

# m3b6 = np.matrix('3 2 1; 1 2 3; 1 2 3; 1 2 3; 1 2 3; 1 2 3')
# m1 = np.matrix('1;2;3')
# thetaT = m1
# k = 1
# theta = 7
# I = quad(left_integrand, 0, 100, args=(thetaT, k, theta))
# print(I)


s0 = .0028930074
s1 = .002575896
s2 = .0022908431
s3 = .0020065713
s4 = .0019838186
s5 = .001785727
s6 = .001622853
s7 = .0014499956
s8 = .0014494543
s9 =.0012997393

g0 = 0.0022
g1 = 0.0025
g2 = 0.0023
g3 = 0.0032
g4 = 0.0031
g5 = 0.0028
g6 = 0.0025
g7 = 0.0026
g8 = 0.003
g9 = 0.0025

t0 = 0.22
t1 = 0.83
t2 = 0.53
t3 = 0.86
t4 = 0.61
t5 = 0.11
t6 = 0.42
t7 = 0.96
t8 = 0.50
t9 = 0.73

T = 100

sigma = np.array([[s0], [s1], [s2], [s3], [s4], 
                  [s5], [s6], [s7], [s8], [s9]]) # mgarch_var_cov_capm.csv
gamma = np.array([[g0], [g1], [g2], [g3], [g4], [g5], [g6], [g7], [g8], [g9]]) # r.csv
theta = np.array([[t0], [t1], [t2], [t3], [t4], [t5], [t6], [t7], [t8], [t9]]) 
K = np.linalg.inv(sigma * sigma.T + gamma * gamma.T)
i1 = T * -0.5 * theta.T * K * theta
left_numerator = np.exp(i1)

w0 = 0.022333875
w1 = 0.275750203
w2 = 0.087193981
w3 = 0.159837832
w4 = 0.180537138
w5 = 0.027291019
w6 = 0.06139176
w7 = 0.077319995
w8 = 0.0541353
w9 = 0.054208916

wb0 = .12825651
wb1 = .12795275
wb2 = .13127413
wb3 = .13269231
wb4 = .13384321
wb5 = .13282733
wb6 = .13307984
wb7 = .13207547
wb8 = .13559322
wb9 = .13721804

wT = np.array([[w0], [w1], [w2], 
       [w3], [w4], [w5], 
       [w6], [w7], [w8], [w9]])

wbT = np.array([[wb0], [wb1], [wb2],
                [wb3], [wb4], [wb5],
                [wb6], [wb7], [wb8], [wb9]])

left_denom = (gamma + theta) * wT
exp1 = left_numerator/left_denom
print(exp1)

# def leftIntegrand(s, thetaT, k, theta):
    # return thetaT*k*theta

# r = quad(leftIntegrand, 0, 100, args=(thetaT, k, theta))

# ---------------------------------------------------------------------------------------

# import pandas as pd
# import scipy as sc
# from numpy import e

# lmd = 1-r.csv
# M, R are matrices with same shape
# def optimum_weight(pi, M, tau, R, eta, lmd, T, theta, t):
    # return pi*M*e**(T*tau - t*tau)/R + eta*(M*e**(T*tau - t*tau)/R
                                            # - 1 + e**(-0.5*T*eta*theta**T +
                                                      # 0.5*eta*t*theta**T)/(R*lmd))


# dates=df['date'].tolist()
# df = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/betas/betas_macro.csv') #betas
# df2 = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/mgarch/mgarch_var_cov_macro.csv') #mgarch
# df3 = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/sig_e_m/macro.csv') #gama_e_m
# dates=df['date'].tolist()


# l=5 ##Numero de columnas modelo
# def betas(df,l):
    # dates=df['date'].tolist()
    # betas={}
    # for i in dates:
        # v=df[df['date']==i].values.tolist()[0]
        # v.pop(0)
        # beta=np.zeros((10,l))
        # for t in range(l*10):
            # r=int(t/10)
            # s=t-10*r
            # beta[s][r]=v[t]
        # betas[i]=beta
    # return betas

# mbetas=betas(df,l)

# def nivel(l,t):
    # # nivel=0
    # for i in range(l):
        # if t<(l*(l+1)/2-((l-i-1)*(l-i)/2)):
            # return i
# def orden(l,t):
    # a=nivel(l,t)
    # return int(t-(l*(l+1)/2)+((l-a)*(l-a+1)/2)+a)


# def mgar(df,l):
    # dates=df['edate'].tolist()
    # mgar={}
    # for i in dates:
        # v=df[df['edate']==i].values.tolist()[0]
        # v.pop(0)
        # gar=np.zeros((l,l))
        # for t in range(int(l*(l+1)/2)):
            # gar[orden(l,t)][nivel(l,t)]=v[t]
            # gar[nivel(l,t)][orden(l,t)]=v[t]
        # mgar[i]=gar
    # return mgar

# mgarvar=mgar(df2,l)


# def multiplex(d):
    # sig=np.zeros((10,10))
    # for i in range(55):
        # index=0
        # for m in reversed(range(0,10)):
            # if 55-i>m*(m+1)/2:
                # index=m
                # break
        # lam=int(11-(55-i-index*(index+1)/2))
        # sig[lam-1][9-index]=d.iloc[i]
        # sig[9-index][lam-1]=d.iloc[i]
    # return sig

# gama_e_m={}
# for i in range(len(dates)):
    # gama_e_m[dates[i]]=multiplex(df3.loc[i])

# gama_m={}
# for i in dates:
    # gama_m[i]=np.matmul(np.matmul(mbetas[i],mgarvar[i[1:-1]]),mbetas[i].T)+gama_e_m[i]

# df4 = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/r.csv')
# df5 = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/w.csv')

# dates3=df5[['crsp_fundno','date']].values.tolist()
# dates2=[]
# dates2=[tuple(x) for x in set(tuple(x) for x in dates3)]
# du1=df4['date'].tolist()
# r_f_t={}
# for i in du1:
    # r_f_t[i]=df4[df4['date']==i]['r_f_t'].tolist()*10



# u_i_m_t={}
# y=0
# for i in dates2:
    # try:
        # u_i_m_t[i]=np.array(r_f_t[i[1][1:-1]])[np.newaxis].T+np.matmul(gama_m[i[1]],np.array(df5[(df5['crsp_fundno']==i[0]) & (df5['date']==i[1])]['weightSIC2'].values.tolist())[np.newaxis].T)
    # except:
        # pass
    # y=y+1
    # if y%50==0:
        # print(y)


# k=pd.Series(u_i_m_t).reset_index()
# k.rename( columns={0:'pi'}, inplace=True )

# def cu(a):
    # p=[]
    # for i in range(len(a)):
        # p.append(a[i][0])
    # return p

# k['pi']=k['pi'].apply(cu)


# k.to_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/results_macro.csv', index = False)

# print(k)
