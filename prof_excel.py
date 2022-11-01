import numpy as np
from openpyxl import load_workbook

wb = load_workbook("rnd.xlsx", read_only=True)
ws = wb.active


# xlsx non-derived matrix grabber (should do it only for samples)
def load(raw):
    arr = np.empty(np.array(raw).shape)
    for i in range(0, np.array(raw).shape[0]):
        for j in range(0, np.array(raw).shape[1]):
            arr[i][j] = raw[i][j].value
    return arr

# grab sample factors
CAPM_factor = ws['b3'].value
FF3_factors = load(ws['b6':'d8'])
MACRO_factors = load(ws['b11':'f15'])

# grab sample w_bar... i.e. Market Weight
w_bar = load(ws['f19':'f28'])

# grab sample w... i.e. Port. Weight
w = load(ws['b19':'b28'])

# grab sample beta capm, ff3 and macro
Beta_CAPM = load(ws['o2':'o11'])
Beta_FF3 = load(ws['o14':'q23'])
Beta_Macro = load(ws['o26':'s35'])

# grab sample sigma sub epsilon, sub t
VC_CAPM_idio = load(ws['ac2':'al11'])
VC_FF3_idio = load(ws['ac14':'al23'])
VC_MACRO_idio = load(ws['ac26':'al35'])

# grab sample get chol (lightgreen)
chol_FF3_VC = load(ws['h6':'j8'])
chol_MACRO_VC = load(ws['h11':'l15'])

# grab idiosyncratics (have formula outside excel)
chol_vc_capm_idio = load(ws['an2':'aw11'])
chol_vc_ff3_idio = load(ws['an14':'aw23'])
chol_vc_macro_idio = load(ws['an26':'aw35'])

# grab sample thetas
theta_FF3 = load(ws['b36':'b45'])
theta_CAPM = load(ws['a36':'a45'])
theta_MACRO = load(ws['c36':'c45'])

# calculate vol_mkt
Vol_Mkt = np.sqrt(CAPM_factor)

# calculate sigma capm, ff3 and macro
Sigma_CAPM_syst = np.dot(Beta_CAPM, Vol_Mkt)
Sigma_FF3_syst = np.dot(Beta_FF3, chol_FF3_VC)
Sigma_MACRO_syst = np.dot(np.dot(Beta_Macro, chol_MACRO_VC), Beta_Macro.T)

# =MMULT(MINVERSE(MMULT(V14:X23,TRANSPOSE(V14:X23))+MMULT(AN14:AW23,TRANSPOSE(AN14:AW23))),B36:B45)
def get_eta(sigma, idio, theta):
    res1, res2 = (np.dot(sigma, sigma.T)+np.dot(idio, idio.T), theta)
    return np.dot(np.linalg.inv(res1), res2)

eta_capm = get_eta(Sigma_CAPM_syst, chol_vc_capm_idio, theta_CAPM) 
eta_ff3 = get_eta(Sigma_FF3_syst, chol_vc_ff3_idio, theta_FF3)
eta_macro = get_eta(Sigma_MACRO_syst, chol_vc_macro_idio, theta_MACRO) #TODO recheck!!

# calculating systematics
VC_CAPM_syst = np.dot(np.dot(Beta_CAPM, CAPM_factor), Beta_CAPM.T)
VC_FF3_syst = np.dot(np.dot(Beta_FF3,  FF3_factors), Beta_FF3.T)

#calculating VC
VC_CAPM = VC_CAPM_syst + VC_CAPM_idio
VC_FF3 = VC_FF3_syst + VC_FF3_idio

# calculating b
b_capm = np.sqrt(np.dot(np.dot(w_bar.T, VC_CAPM), w_bar))
# b_ff3 = np.sqrt(np.dot(np.dot(w_bar.T, VC_FF3), w_bar)) # TODO comeback later
b_ff3 = np.array([[0.034], [0], [0]])
# b_macro =  ws['ba26'].value
# b_macro = np.array([[0.034], [0], [0], [0], [0]])

# =MMULT(MINVERSE(MMULT(V14:X23,TRANSPOSE(V14:X23))+MMULT(AN14:AW23,TRANSPOSE(AN14:AW23))),
# MMULT(V14:X23,BA14:BA16))
def get_pi(sigma, idio, beta):
    res1, res2 = (np.dot(sigma, sigma.T)+np.dot(idio, idio.T), np.dot(sigma,beta))
    return np.dot(np.linalg.inv(res1), res2)

pi_capm = get_pi(Sigma_CAPM_syst, chol_vc_capm_idio, b_capm)
pi_ff3 = get_pi(Sigma_FF3_syst, chol_vc_ff3_idio, b_ff3)
print(pi_ff3)
# pi_macro = get_pi(Sigma_MACRO_syst, chol_vc_macro_idio, b_macro)
