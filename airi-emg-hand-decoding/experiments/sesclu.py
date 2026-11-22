import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from common import *
meta,xs,ys=load_all()
def sf(x):
    x=x/1e-5
    env=uniform_filter1d(np.abs(x),100,axis=0)
    fl=np.log(np.percentile(env,5,0))
    fr,p=welch(x,fs=500,nperseg=1024,axis=0); lp=np.log(p)
    hi=lp[(fr>200)].mean(0); mid=lp[(fr>40)&(fr<120)].mean(0)
    shape=(lp[(fr>5)]-lp[(fr>5)].mean(0)).reshape(-1,6)
    shape=np.stack([shape[i::1][:].mean(0) for i in range(1)]).ravel()
    lpb=lp[fr>2]; nb=20; bands=np.array_split(lpb,nb); bs=np.stack([b.mean(0) for b in bands]); bs=bs-bs.mean(0)
    c=np.corrcoef(x.T)[np.triu_indices(6,1)]
    line=lp[(fr>49)&(fr<51)].max(0)-lp[(fr>45)&(fr<55)].mean(0)
    return dict(floor=fl-fl.mean(), floorabs=fl, hi=hi-hi.mean(), corr=c, bandshape=bs.ravel(), line=line, mean=x.mean(0)*100)
D=[sf(xs[r]) for r in meta.recording_id]
lab=meta.session_id.astype('category').cat.codes.values
for keys in [['floor'],['floorabs'],['hi'],['corr'],['bandshape'],['line'],['mean'],['floor','corr','bandshape'],['floorabs','corr','bandshape','hi']]:
    F=np.stack([np.concatenate([d[k] for k in keys]) for d in D]); F=StandardScaler().fit_transform(F)
    res=[]
    for seed in range(5):
        rng=np.random.default_rng(seed); ses=rng.choice(meta.session_id.unique(),4,replace=False)
        idx=meta.index[meta.session_id.isin(ses)]
        p=AgglomerativeClustering(4,linkage='ward').fit_predict(F[idx]); res.append(adjusted_rand_score(lab[idx],p))
    print(keys,'ARI(4 random sessions)',np.round(res,2))
