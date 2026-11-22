import numpy as np, sys, pandas as pd
from common import *
from nn import upsample
meta,xs,ys=load_all()
yl=meta.action.map({a:i for i,a in enumerate(ACTIONS)}).values
O=np.mean([np.load(f)[np.arange(80),yl] for f in sys.argv[1:]],0)
ses=meta.session_id.values; day=meta.day_id.values
def errs(pred):
    return np.array([record_error(ys[r],upsample(pred[i][:(len(ys[r])+9)//10],len(ys[r])),a) for i,(r,a) in enumerate(zip(meta.recording_id,meta.action))])
def sc(e): return np.mean([e[ses==s].mean() for s in np.unique(ses)])
e0=errs(O); print('base',round(sc(e0),4))
for act in ['Open','Close']:
    for alpha in [1.0,0.7,0.5,0.0]:
        P=O.copy()
        for i in np.where(meta.action==act)[0]:
            trm=np.median(np.concatenate([ys[r][1000:] for r in meta.recording_id[(meta.action==act)&(day!=day[i])]]),0)
            const=np.median(O[i][100:],0)
            P[i]=alpha*const+(1-alpha)*trm
        print(act,'const alpha',alpha,round(sc(errs(P)),4))
for shrink in [1.0,0.5]:
    P=O.copy()
    for act in ACTIONS:
        idx=np.where(meta.action==act)[0]
        for i in idx:
            tr=[j for j in idx if day[j]!=day[i]]
            yp=np.concatenate([O[j][100:2050] for j in tr]); yt=np.concatenate([ys[meta.recording_id[j]][::10][100:2050] for j in tr])
            m=np.median(yp,0); a=np.ones(20); b=np.zeros(20)
            for k in range(20):
                best=(np.abs(yt[:,k]-yp[:,k]).mean(),1,0)
                for aa in [0.8,0.9,1.1,1.2,1.3,1.5]:
                    r=yt[:,k]-(aa*(yp[:,k]-m[k])+m[k]); bb=np.median(r); v=np.abs(r-bb).mean()
                    if v<best[0]: best=(v,aa,bb)
                a[k]=1+shrink*(best[1]-1); b[k]=shrink*best[2]
            P[i]=a*(O[i]-m)+m+b
    print('affine calib shrink',shrink,round(sc(errs(P)),4))
