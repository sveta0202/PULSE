import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from scipy.optimize import linear_sum_assignment
from common import *
meta,xs,ys=load_all()
def rec_feats(x):
    x=x[:]/1e-5
    env=uniform_filter1d(np.abs(x),100,axis=0)
    le=np.log(env+1e-3)
    m=le.mean(0); f=[m-m.mean(), [m.mean()], le.std(0), np.percentile(le,90,0)-np.percentile(le,10,0)]
    c=np.corrcoef(le.T)[np.triu_indices(6,1)]; f.append(c)
    s=le.mean(1); s=s-s.mean(); ac=np.correlate(s[::10],s[::10],'full')[len(s[::10])-1:]; ac/=ac[0]
    f.append(ac[[25,50,65,80,100,130,200]])
    fr,p=welch(x,fs=500,nperseg=512,axis=0); lp=np.log(p+1e-12)
    bands=[(5,30),(30,60),(60,100),(100,150),(150,250)]
    bp=np.stack([lp[(fr>=a)&(fr<b)].mean(0) for a,b in bands]); f.append((bp-bp.mean(0)).ravel())
    cov=np.corrcoef(x.T)[np.triu_indices(6,1)]; f.append(cov)
    return np.concatenate([np.ravel(v) for v in f])
F=np.stack([rec_feats(xs[r]) for r in meta.recording_id]); yl=meta.action.map({a:i for i,a in enumerate(ACTIONS)}).values
for name,mk in [('lr',lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=3000))),('et',lambda: ExtraTreesClassifier(500,random_state=0))]:
    P=np.zeros((len(meta),8))
    for tr,va in day_folds(meta):
        m=mk().fit(F[tr.index],yl[tr.index]); P[va.index]=m.predict_proba(F[va.index])
    acc=(P.argmax(1)==yl).mean()
    hung=[]
    for s,g in meta.groupby('session_id'):
        ri,ci=linear_sum_assignment(-np.log(P[g.index]+1e-9)); hung+=list(ci==yl[g.index][ri])
    print(name,'acc',acc,'hungarian-per-session acc',np.mean(hung))
    print(pd.crosstab(meta.action,np.array(ACTIONS)[P.argmax(1)]))
print('--- balanced assignment over 4 random sessions (lr probs)')
m=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=3000))
P=np.zeros((len(meta),8))
for tr,va in day_folds(meta):
    P[va.index]=m().fit(F[tr.index],yl[tr.index]).predict_proba(F[va.index])
np.save('outputs/clf_cv_probs.npy',P)
accs=[];accb=[]
sess=meta.session_id.unique()
for seed in range(200):
    rng=np.random.default_rng(seed); ses=rng.choice(sess,4,replace=False); idx=meta.index[meta.session_id.isin(ses)].values
    C=-np.log(P[idx]+1e-9); C=np.repeat(C,4,axis=1)
    ri,ci=linear_sum_assignment(C); accb.append(np.mean(ci//4==yl[idx][ri])); accs.append(np.mean(P[idx].argmax(1)==yl[idx]))
print('argmax',np.mean(accs),'balanced',np.mean(accb))
for C_ in [0.03,0.1,0.3,1,3]:
    P2=np.zeros((len(meta),8))
    for tr,va in day_folds(meta):
        P2[va.index]=make_pipeline(StandardScaler(),LogisticRegression(C=C_,max_iter=3000)).fit(F[tr.index],yl[tr.index]).predict_proba(F[va.index])
    print(C_, (P2.argmax(1)==yl).mean(), 'logloss', -np.log(P2[np.arange(80),yl]).mean())
