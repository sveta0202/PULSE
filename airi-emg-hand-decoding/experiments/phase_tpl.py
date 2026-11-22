import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from common import *
from nn import upsample
meta,xs,ys=load_all()
yl=meta.action.map({a:i for i,a in enumerate(ACTIONS)}).values
O=np.load('outputs/ens_oof_v2.npy')[np.arange(80),yl]
ses=meta.session_id.values
def sc(P):
    e=np.array([record_error(ys[r],upsample(P[i][:(len(ys[r])+9)//10],len(ys[r])),a) for i,(r,a) in enumerate(zip(meta.recording_id,meta.action))])
    return np.mean([e[ses==s].mean() for s in np.unique(ses)])
def tpl(p, nb, lo, hi):
    pc=p-p.mean(0); _,_,vt=np.linalg.svd(pc[100:],full_matrices=False); s=pc@vt[0]
    s=sosfiltfilt(butter(2,[lo,hi],btype='band',fs=50,output='sos'),s)
    ph=np.angle(hilbert(s)); b=((ph+np.pi)/(2*np.pi)*nb).astype(int)%nb
    m=np.stack([np.median(p[100:][b[100:]==k],0) if (b[100:]==k).sum()>5 else p[100:].mean(0) for k in range(nb)])
    return m[b]
print('base',round(sc(O),4))
cyc=[a not in ('Open','Close') for a in meta.action]
for nb in [24,48]:
  for lo,hi in [(0.3,1.5),(0.4,1.2)]:
    T=np.stack([tpl(O[i][:2051],nb,lo,hi) if cyc[i] else O[i][:2051] for i in range(80)])
    for w in [0.9,0.8,0.6]:
        print(nb,lo,hi,w,round(sc(w*O[:, :2051]+(1-w)*T),4))
