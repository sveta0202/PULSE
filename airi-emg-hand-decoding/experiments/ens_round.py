import numpy as np, sys, glob, itertools
from common import *
from nn import upsample
meta,xs,ys=load_all()
yl=meta.action.map({a:i for i,a in enumerate(ACTIONS)}).values
ses=meta.session_id.values; us=np.unique(ses)
Ys=[ys[r][::10][:2051] for r in meta.recording_id]
def sc(P):
    e=[]
    for i,a in enumerate(meta.action):
        y=Ys[i]; n=len(y); err=np.abs(y[100:n]-P[i][100:n]).reshape(-1,5,4).mean(axis=(0,2)); k=ACTIVE.get(a)
        e.append(err.mean() if k is None else 0.5*err[k]+0.125*(err.sum()-err[k]))
    e=np.array(e); return np.mean([e[ses==s].mean() for s in us])
tags=sys.argv[1].split(',')
O={t:np.load(f'outputs/oof_{t}.npy',mmap_mode='r')[np.arange(80),yl][:, :2051] for t in tags}
ens=np.load('outputs/ens_oof_v2.npy',mmap_mode='r')[np.arange(80),yl][:, :2051]
for t in tags: print(t, round(sc(O[t]),4))
sel=[];
for it in range(8):
    best=min(((sc(np.mean([O[j] for j in sel+[k]],0)),k) for k in tags))
    sel.append(best[1]); print('add',best[1],round(best[0],4))
S=np.mean([O[t] for t in tags],0)
print('all students',round(sc(S),4))
for w in [0,0.2,0.3,0.5]: print('mix teacher',w,round(sc(w*ens+(1-w)*S),4))
