import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from common import *
meta,xs,ys=load_all()
def sf(x):
    x=x/1e-5
    fr,p=welch(x,fs=500,nperseg=1024,axis=0); lp=np.log(p)
    lpb=lp[fr>2]; bands=np.array_split(lpb,24); bs=np.stack([b.mean(0) for b in bands])
    env=uniform_filter1d(np.abs(x),100,axis=0); le=np.log(env)
    q=np.percentile(le,[2,5,10,25,50],0)
    c=np.corrcoef(x.T)[np.triu_indices(6,1)]
    ce=np.corrcoef(le.T)[np.triu_indices(6,1)]
    return dict(bs=bs.ravel(), bsrel=(bs-bs.mean(1,keepdims=True)).ravel(), q=q.ravel(), qrel=(q-q.mean(1,keepdims=True)).ravel(), c=c, ce=ce)
D=[sf(xs[r]) for r in meta.recording_id]
ses=meta.session_id.values; act=meta.action.values; day=meta.day_id.values
def evalset(keys, ncomp=6, shrink='auto'):
    F=np.stack([np.concatenate([d[k] for k in keys]) for d in D])
    aucs=[]
    days=np.unique(day); rng=np.random.default_rng(0)
    for it in range(15):
        hold=rng.choice(days,3,replace=False); te=np.isin(day,hold); tr=~te
        if len(np.unique(ses[te]))<2: continue
        sc=StandardScaler().fit(F[tr]); Ft=sc.transform(F)
        lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage=shrink).fit(Ft[tr],ses[tr]); Z=lda.transform(Ft)[:, :ncomp]
        idx=np.where(te)[0]; ds=[];lab=[]
        for i in idx:
            for j in idx:
                if i<j and act[i]!=act[j]: ds.append(-np.linalg.norm(Z[i]-Z[j])); lab.append(ses[i]==ses[j])
        aucs.append(roc_auc_score(lab,ds))
    return np.mean(aucs)
for keys in [['bs'],['bsrel'],['q'],['qrel'],['c'],['ce'],['bs','c'],['bsrel','qrel','c'],['bs','q','c','ce']]:
    print(keys,[round(evalset(keys,n),3) for n in (3,6,9)])
