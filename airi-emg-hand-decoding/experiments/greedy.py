import numpy as np, glob
from common import *
from nn import upsample
meta,xs,ys=load_all()
yl=meta.action.map({a:i for i,a in enumerate(ACTIONS)}).values
files=sorted(glob.glob('outputs/oof_*.npy'))
O={f.split('oof_')[1][:-4]:np.load(f)[np.arange(80),yl] for f in files}
O={k:v for k,v in O.items() if 'ema999' not in k}
ses=meta.session_id.values; us=np.unique(ses)
Ys=[ys[r][::10] for r in meta.recording_id]
def sc(P):
    e=[]
    for i,(r,a) in enumerate(zip(meta.recording_id,meta.action)):
        y=Ys[i]; n=min(len(y),2050)
        err=np.abs(y[100:n]-P[i][100:n]).reshape(-1,5,4).mean(axis=(0,2)); k=ACTIVE.get(a)
        e.append(err.mean() if k is None else 0.5*err[k]+0.125*(err.sum()-err[k]))
    e=np.array(e); return np.mean([e[ses==s].mean() for s in us])
for k,v in sorted(O.items(), key=lambda kv: sc(kv[1])): print(k, round(sc(v),4))
sel=[];cur=None
for it in range(8):
    best=None
    for k in O:
        P=np.mean([O[j] for j in sel+[k]],0); s=sc(P)
        if best is None or s<best[0]: best=(s,k)
    sel.append(best[1]); print('add',best[1],round(best[0],4))
