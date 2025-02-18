import pandas as pd
import numpy as np
#df = pd.read_csv(r'C:\Users\soporte\Desktop\Finance_upwork_job_paper\betas\betas_macro.csv') #betas
#df2=pd.read_csv(r'C:\Users\soporte\Desktop\Finance_upwork_job_paper\mgarch\mgarch_var_cov_macro.csv') #mgarch
#df3 = pd.read_csv(r'C:\Users\soporte\Desktop\Finance_upwork_job_paper\sig_e_m\macro.csv') #gama_e_m
df = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/betas/betas_macro.csv') #betas
df2 = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/mgarch/mgarch_var_cov_macro.csv') #mgarch
df3 = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/sig_e_m/macro.csv') #gama_e_m
dates=df['date'].tolist()


l=5 ##Numero de columnas modelo
def betas(df,l):
    dates=df['date'].tolist()
    betas={}
    for i in dates:
        v=df[df['date']==i].values.tolist()[0]
        v.pop(0)
        beta=np.zeros((10,l))
        for t in range(l*10):
            r=int(t/10)
            s=t-10*r
            beta[s][r]=v[t]
        betas[i]=beta
    return betas

mbetas=betas(df,l)

def nivel(l,t):
    # nivel=0
    for i in range(l):
        if t<(l*(l+1)/2-((l-i-1)*(l-i)/2)):
            return i
def orden(l,t):
    a=nivel(l,t)
    return int(t-(l*(l+1)/2)+((l-a)*(l-a+1)/2)+a)


def mgar(df,l):
    dates=df['edate'].tolist()
    mgar={}
    for i in dates:
        v=df[df['edate']==i].values.tolist()[0]
        v.pop(0)
        gar=np.zeros((l,l))
        for t in range(int(l*(l+1)/2)):
            gar[orden(l,t)][nivel(l,t)]=v[t]
            gar[nivel(l,t)][orden(l,t)]=v[t]
        mgar[i]=gar
    return mgar

mgarvar=mgar(df2,l)


def multiplex(d):
    sig=np.zeros((10,10))
    for i in range(55):
        index=0
        for m in reversed(range(0,10)):
            if 55-i>m*(m+1)/2:
                index=m
                break
        lam=int(11-(55-i-index*(index+1)/2))
        sig[lam-1][9-index]=d.iloc[i]
        sig[9-index][lam-1]=d.iloc[i]
    return sig

gama_e_m={}
for i in range(len(dates)):
    gama_e_m[dates[i]]=multiplex(df3.loc[i])

gama_m={}
for i in dates:
    gama_m[i]=np.matmul(np.matmul(mbetas[i],mgarvar[i[1:-1]]),mbetas[i].T)+gama_e_m[i]

# df4 = pd.read_csv(r'C:\Users\soporte\Desktop\Finance_upwork_job_paper\r.csv')
# df5 = pd.read_csv(r'C:\Users\soporte\Desktop\Finance_upwork_job_paper\w.csv')
df4 = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/r.csv')
df5 = pd.read_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/w.csv')

dates3=df5[['crsp_fundno','date']].values.tolist()
dates2=[]
dates2=[tuple(x) for x in set(tuple(x) for x in dates3)]
du1=df4['date'].tolist()
r_f_t={}
for i in du1:
    r_f_t[i]=df4[df4['date']==i]['r_f_t'].tolist()*10



u_i_m_t={}
y=0
for i in dates2:
    try:
        u_i_m_t[i]=np.array(r_f_t[i[1][1:-1]])[np.newaxis].T+np.matmul(gama_m[i[1]],np.array(df5[(df5['crsp_fundno']==i[0]) & (df5['date']==i[1])]['weightSIC2'].values.tolist())[np.newaxis].T)
    except:
        pass
    y=y+1
    if y%50==0:
        print(y)


k=pd.Series(u_i_m_t).reset_index()
k.rename( columns={0:'pi'}, inplace=True )

def cu(a):
    p=[]
    for i in range(len(a)):
        p.append(a[i][0])
    return p

k['pi']=k['pi'].apply(cu)


# k.to_csv(r'C:\Users\soporte\Desktop\Finance_upwork_job_paper\results_macro.csv', index = False)
k.to_csv(r'/home/saifr/rnd/Finance_upwork_job_paper/results_macro.csv', index = False)

print(k)
