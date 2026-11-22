import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
meta,xs,ys=load_all()
yl=meta.action.map({a:i for i,a in enumerate(ACTIONS)}).values
O=np.load('outputs/oof_r4_ref.npy')[np.arange(80),yl]
cases=[('ses_c2c7265f','Index'),('ses_c2c7265f','Ring'),('ses_fa6b7158','Fist'),('ses_0ca10a67','Index'),('ses_75a236ea','Little'),('ses_0ca10a67','Thumb')]
fig,ax=plt.subplots(len(cases),1,figsize=(18,4*len(cases)))
for k,(s,a) in enumerate(cases):
    i=meta.index[(meta.session_id==s)&(meta.action==a)][0]; r=meta.recording_id[i]
    f=ACTIVE.get(a,0); y=ys[r][::10]
    for j in range(4):
        ax[k].plot(y[:,f*4+j],c=f'C{j}',lw=1); ax[k].plot(O[i][:,f*4+j],c=f'C{j}',ls='--',lw=1)
    ax[k].set_title(f'{s} {a} finger {f}: solid=true dashed=pred'); ax[k].set_xlim(100,1300)
plt.tight_layout(); plt.savefig('outputs/pred_vs_true.png',dpi=45)
