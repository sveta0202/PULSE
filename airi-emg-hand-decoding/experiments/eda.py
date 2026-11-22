import numpy as np, pandas as pd
from common import *
meta, xs, ys = load_all()
def cv(fn):
    preds={}
    for tr,va in day_folds(meta):
        for r in va.itertuples(index=False): preds[r.recording_id]=fn(tr,r)
    s,df=score(meta,preds,ys); return s,df
def stack(rows): return np.concatenate([ys[r][CONTEXT:] for r in rows.recording_id])
glob_med=lambda tr,r: np.tile(np.median(stack(tr),0),(len(ys[r.recording_id]),1))
act_med=lambda tr,r: np.tile(np.median(stack(tr[tr.action==r.action]),0),(len(ys[r.recording_id]),1))
for name,fn in [('global median',glob_med),('oracle action median',act_med)]:
    s,df=cv(fn); print(name, round(s,4)); print(df.groupby('action').err.mean().round(3).to_dict())
own=lambda tr,r: np.tile(np.median(ys[r.recording_id][CONTEXT:],0),(len(ys[r.recording_id]),1))
s,df=cv(own); print('own-recording median (oracle static)',round(s,4)); print(df.groupby('action').err.mean().round(3).to_dict())
for a in ACTIONS:
    sub=meta[meta.action==a]
    per=np.stack([np.median(ys[r][CONTEXT:],0) for r in sub.recording_id])
    stdt=np.mean([ys[r][CONTEXT:].std(0).mean() for r in sub.recording_id])
    print(a,'between-session std of median pose',per.std(0).mean().round(3),'within-rec temporal std',round(stdt,3))
x=np.concatenate([xs[r] for r in meta.recording_id]); print('emg std per ch',x.std(0))
for s_,g in meta.groupby('session_id'): print(s_, np.concatenate([xs[r] for r in g.recording_id]).std(0).round(7))
