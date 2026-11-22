import sys, json, numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from common import *
from feats import envelopes, lagstack
from nn import upsample
meta,xs,ys=load_all()
E={r:envelopes(xs[r]) for r in meta.recording_id}
Yd={r:ys[r][::10] for r in meta.recording_id}
def run(span,step,K,ncomp,center):
    lags=list(range(-span,span+1,step))
    F={r:lagstack(E[r]-(np.median(E[r],0) if center else 0),lags) for r in meta.recording_id}
    preds={}
    for tr,va in day_folds(meta):
        for act in ACTIONS:
            trr=tr[tr.action==act].recording_id
            X=np.concatenate([F[r][100:] for r in trr]); Y=np.concatenate([Yd[r][100:] for r in trr])
            mu,sd=X.mean(0),X.std(0)+1e-6; pca=PCA(ncomp,random_state=0).fit((X-mu)/sd); Z=pca.transform((X-mu)/sd)
            nnb=NearestNeighbors(n_neighbors=K).fit(Z)
            for r in va[va.action==act].recording_id:
                _,idx=nnb.kneighbors(pca.transform((F[r]-mu)/sd)); p=np.median(Y[idx],1)
                preds[r]=upsample(p,len(ys[r]))
    s,df=score(meta,preds,ys); return s,df.groupby('action').err.mean().round(3).to_dict()
for cfg in [(40,4,20,20,0),(40,4,50,20,0),(20,2,20,20,0),(40,4,20,10,0),(40,4,20,20,1)]:
    print(cfg, *run(*cfg), flush=True)
