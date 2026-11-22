import sys, numpy as np, pandas as pd
from common import *
meta,xs,ys=load_all(); sub=pd.read_csv(sys.argv[1]); cols=[f'angle_{i:02d}' for i in range(20)]
act=pd.read_csv(sys.argv[1].replace('.csv','.actions.csv'))
preds={}
for r,g in sub.groupby('recording_id'):
    p=np.zeros((len(ys[r]),20),np.float32); p[g.sample_index.values]=g[cols].values; preds[r]=p
m=meta[meta.recording_id.isin(preds)]
s,df=score(m,preds,ys); print('holdout wMAE',round(s,4)); print(df.groupby('action').err.mean().round(3).to_dict())
acc=(m.set_index('recording_id').action.loc[act.recording_id].values==act.action.values).mean(); print('action acc',acc)
