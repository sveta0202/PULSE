import numpy as np, itertools
exec(open('experiments/greedy.py').read().split("for k,v in sorted")[0])
fam={'gru_env':['g05_seeds3','r4_ref','r4_chd','r5_pseudo1','r5_pseudo1_w05'],
     'lstm_env':['r8_lstm','r11_lstm_chd','r11_lstm_pseudo','r11_lstm_h96','r11_lstm_s1500','r17_lstm_ref_s3'],
     'lstm_new':['r17_lstm_envd_s3','r17_lstm_envd_ssl500','r17_lstm_ssl500_s3','r15_lstm_ssl1500','r15_lstm_ssl500','r14_lstm_envd'],
     'corr':['r5_envcorr','r5_envcorr_chd','r7_envcorr_pseudo'],
     'tan':['r16_tan_gru','r16_tan_lstm'],
     'raw':['r9_raw_chd','r10_raw_chd3','r10_raw_chd_pseudo','r10_tf_raw_chd','r15_lstm_raw_ssl1000'],
     'lstm_raw':['r8_lstm_raw','r11_lstm_raw_chd','r13_lstm_raw_s1500']}
F={k:np.mean([O[j] for j in v],0) for k,v in fam.items()}
for k in F: print(k, round(sc(F[k]),4))
R=[]
for w in itertools.product([0,1,2,3],repeat=len(F)):
    if sum(w)==0: continue
    P=sum(wi*F[k] for wi,k in zip(w,F))/sum(w); s=sc(P)
    R.append((s,w))
R.sort(); [print(dict(zip(F,w)), round(s,4)) for s,w in R[:12]]
