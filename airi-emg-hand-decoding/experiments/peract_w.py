import numpy as np, itertools
from common import *
from nn import upsample
meta,xs,ys=load_all()
yl=meta.action.map({a:i for i,a in enumerate(ACTIONS)}).values
fam={'gru_env':['g05_seeds3','r4_ref','r4_chd','r5_pseudo1','r5_pseudo1_w05'],
     'lstm_env':['r8_lstm','r11_lstm_chd','r11_lstm_pseudo','r11_lstm_h96','r11_lstm_s1500','r17_lstm_ref_s3'],
     'lstm_new':['r17_lstm_envd_s3','r17_lstm_envd_ssl500','r17_lstm_ssl500_s3','r15_lstm_ssl1500','r15_lstm_ssl500','r14_lstm_envd'],
     'corr':['r5_envcorr','r5_envcorr_chd','r7_envcorr_pseudo'],
     'tan':['r16_tan_gru','r16_tan_lstm'],
     'raw':['r9_raw_chd','r10_raw_chd3','r10_raw_chd_pseudo','r10_tf_raw_chd','r15_lstm_raw_ssl1000'],
     'lstm_raw':['r8_lstm_raw','r11_lstm_raw_chd','r13_lstm_raw_s1500']}
F={k:np.mean([np.load(f'outputs/oof_{t}.npy',mmap_mode='r')[np.arange(80),yl][:, :2051] for t in v],0) for k,v in fam.items()}
names=list(F); Fs=np.stack([F[k] for k in names])
Ys=[ys[r][::10][:2051] for r in meta.recording_id]
def rec_err(i,P):
    y=Ys[i]; n=len(y); err=np.abs(y[100:n]-P[100:n]).reshape(-1,5,4).mean(axis=(0,2)); k=ACTIVE.get(meta.action[i])
    return err.mean() if k is None else 0.5*err[k]+0.125*(err.sum()-err[k])
grid=[w for w in itertools.product([0,1,2,3],repeat=len(names)) if sum(w)>0]
gw=np.array([1,1,3,1,2,1,3],float)
ses=meta.session_id.values; us=np.unique(ses)
E=np.zeros((80,len(grid)))
for gi,w in enumerate(grid):
    w=np.array(w,float); P=np.tensordot(w/w.sum(),Fs,1)
    for i in range(80): E[i,gi]=rec_err(i,P[i])
base=np.array([rec_err(i,np.tensordot(gw/gw.sum(),Fs[:,i],1)) for i in range(80)])
chosen=np.zeros(80)
for s in us:
    for a in ACTIONS:
        trn=(meta.action.values==a)&(ses!=s); te=np.where((meta.action.values==a)&(ses==s))[0]
        g=E[trn].mean(0).argmin(); chosen[te]=E[te,g]
sc=lambda e: np.mean([e[ses==s].mean() for s in us])
print('global weights',round(sc(base),4),'per-action weights (leave-session-out honest)',round(sc(chosen),4))
for a in ACTIONS:
    m=meta.action.values==a; print(a, dict(zip(names,grid[E[m].mean(0).argmin()])), round(E[m].mean(0).min(),4), round(base[m].mean(),4))
