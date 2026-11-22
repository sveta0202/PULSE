import sys
from pathlib import Path
import numpy as np, pandas as pd
from common import *
days=sys.argv[1].split(','); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
meta=load_meta(); h=meta[meta.day_id.isin(days)].sample(frac=1,random_state=0)
rows=[]
for r in h.recording_id:
    x=np.load(DATA/'train'/'signals'/f'{r}.npy'); np.save(out/f'{r}.npy',x)
    for t in range(1000,len(x)): rows.append((r,t))
df=pd.DataFrame(rows,columns=['recording_id','sample_index']); df.insert(0,'row_id',range(len(df)))
for i in range(20): df[f'angle_{i:02d}']=0.0
df.to_csv(out.parent/(out.name+'_template.csv'),index=False); print(len(h),'recs',h.session_id.nunique(),'sessions', len(df),'rows')
