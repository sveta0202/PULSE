import numpy as np, pandas as pd, sys, itertools
from sklearn.linear_model import Ridge
from common import *
from feats import *
meta,xs,ys=load_all()
E={r:envelopes(xs[r]) for r in meta.recording_id}
def up(p,T): return p[np.minimum(np.arange(T)//DS,len(p)-1)]
def feat(r,mode,lags):
    f=E[r]
    if mode=='rec': f=norm_rec(f)
    elif mode=='both': f=np.concatenate([f,norm_rec(f)],1)
    elif mode=='center': f=f-np.median(f,0)
    return lagstack(f,lags)
def run(mode,lags,alpha):
    preds={}
    for tr,va in day_folds(meta):
        for a in ACTIONS:
            trr=tr[tr.action==a].recording_id; var=va[va.action==a].recording_id
            X=np.concatenate([feat(r,mode,lags)[100:] for r in trr]); Y=np.concatenate([ys[r][::DS][100:] for r in trr])
            mu,sd=X.mean(0),X.std(0)+1e-6
            m=Ridge(alpha=alpha).fit((X-mu)/sd,Y)
            for r in var: preds[r]=up(m.predict((feat(r,mode,lags)-mu)/sd),len(ys[r]))
    return score(meta,preds,ys)
for mode,(span,step),alpha in itertools.product(['none','both','center'],[(40,4),(80,8),(20,2)],[1000,5000,20000]):
    s,df=run(mode,list(range(-span,span+1,step)),alpha)
    print(mode,span,step,alpha,round(s,4),df.groupby('action').err.mean().round(3).to_dict(),flush=True)
