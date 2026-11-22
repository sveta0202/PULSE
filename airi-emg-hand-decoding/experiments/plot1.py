import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from common import *
meta,xs,ys=load_all()
ses=meta.session_id.unique()[0]
fig,ax=plt.subplots(8,2,figsize=(18,22))
for i,a in enumerate(ACTIONS):
    r=meta[(meta.session_id==ses)&(meta.action==a)].recording_id.iloc[0]
    x=xs[r]; y=ys[r]
    env=uniform_filter1d(np.abs(x),100,axis=0)
    ax[i,0].plot(env/1e-5,lw=.6); ax[i,0].set_title(a+' EMG env')
    ax[i,1].plot(y,lw=.6); ax[i,1].set_title(a+' angles')
plt.tight_layout(); plt.savefig('outputs/eda_session.png',dpi=50)
