import numpy as np, sys
sys.path.insert(0,'reference')
from organizer_ridge_pipeline import features
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import *
meta,xs,ys=load_all()
F={r:features(xs[r]) for r in meta.recording_id}
for alpha in [100,1000,10000]:
    preds={}
    for tr,va in day_folds(meta):
        X=np.concatenate([F[r][1000::20] for r in tr.recording_id]); Y=np.concatenate([ys[r][1000::20] for r in tr.recording_id])
        m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(X,Y)
        for r in va.recording_id: preds[r]=m.predict(F[r])
    s,df=score(meta,preds,ys); print(alpha,round(s,4),df.groupby('action').err.mean().round(3).to_dict())
