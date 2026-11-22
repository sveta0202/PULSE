import numpy as np, itertools
from scipy.optimize import linear_sum_assignment
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from common import *
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
def sf(x):
    x=x/1e-5
    fr,p=welch(x,fs=500,nperseg=1024,axis=0); lp=np.log(p)
    lpb=lp[fr>2]; bands=np.array_split(lpb,24); bs=np.stack([b.mean(0) for b in bands])
    env=uniform_filter1d(np.abs(x),100,axis=0); le=np.log(env)
    q=np.percentile(le,[2,5,10,25,50],0)
    c=np.corrcoef(x.T)[np.triu_indices(6,1)]
    ce=np.corrcoef(le.T)[np.triu_indices(6,1)]
    return dict(bs=bs.ravel(), bsrel=(bs-bs.mean(1,keepdims=True)).ravel(), q=q.ravel(), qrel=(q-q.mean(1,keepdims=True)).ravel(), c=c, ce=ce)
meta,xs,ys=load_all()
D=[sf(xs[r]) for r in meta.recording_id]
ses=meta.session_id.values; act=meta.action.values; day=meta.day_id.values
def feats(keys): return np.stack([np.concatenate([d[k] for k in keys]) for d in D])
def partition(Z, acts):
    groups=[[i] for i in np.where(acts=='Close')[0]]
    order=[a for a in ACTIONS if a!='Close']
    for a in order:
        idx=np.where(acts==a)[0]
        C=np.array([[np.mean([np.linalg.norm(Z[i]-Z[j]) for j in g]) for g in groups] for i in idx])
        ri,ci=linear_sum_assignment(C)
        for r,c in zip(ri,ci): groups[c].append(idx[r])
    for it in range(5):
        changed=False
        for a in ACTIONS:
            idx=[next(i for i in g if acts[i]==a) for g in groups]
            C=np.array([[np.mean([np.linalg.norm(Z[i]-Z[j]) for j in g if acts[j]!=a]) for g in groups] for i in idx])
            ri,ci=linear_sum_assignment(C)
            if (ci!=np.arange(4)).any():
                changed=True
                rest=[[j for j in g if acts[j]!=a] for g in groups]
                for r,c in zip(ri,ci): rest[c].append(idx[r])
                groups=rest
        if not changed: break
    lab=np.zeros(len(acts),int)
    for k,g in enumerate(groups): lab[g]=k
    return lab
from sklearn.metrics import adjusted_rand_score
us=np.unique(ses)
for keys in [['qrel','c'],['bsrel','qrel','c'],['bs','q','c'],['bsrel','qrel','c','ce']]:
    F=feats(keys); aris=[]; exact=[]
    rng=np.random.default_rng(0)
    for t in range(40):
        pick=rng.choice(us,4,replace=False); te=np.isin(ses,pick); trm=~np.isin(day,np.unique(day[te]))
        if len(np.unique(ses[trm]))<3: continue
        sc=StandardScaler().fit(F[trm]); Ft=sc.transform(F)
        lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto').fit(Ft[trm],ses[trm]); Z=lda.transform(Ft)
        idx=np.where(te)[0]; lab=partition(Z[idx],act[idx])
        aris.append(adjusted_rand_score(ses[idx],lab))
        exact.append(np.mean([len(set(ses[idx][lab==k]))==1 for k in range(4)]))
    print(keys,'ARI',round(np.mean(aris),3),'pure groups frac',round(np.mean(exact),3), 'n',len(aris))
