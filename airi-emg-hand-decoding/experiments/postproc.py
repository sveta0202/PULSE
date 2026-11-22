import numpy as np, sys
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import median_filter
from common import *
from nn import upsample
meta,xs,ys=load_all()
yl=meta.action.map({a:i for i,a in enumerate(ACTIONS)}).values
files=sys.argv[1:]
O=[np.load(f)[np.arange(80),yl] for f in files]
ses=meta.session_id.values
def sc(pred):
    e=np.array([record_error(ys[r],upsample(pred[i][:(len(ys[r])+9)//10],len(ys[r])),a) for i,(r,a) in enumerate(zip(meta.recording_id,meta.action))])
    return np.mean([e[ses==s].mean() for s in np.unique(ses)]), e
mean=np.mean(O,0); med=np.median(O,0)
print('mean',round(sc(mean)[0],4),'median',round(sc(med)[0],4))
for c in [1,2,3,5]:
    sos=butter(2,c,fs=50,output='sos'); print('lowpass',c,round(sc(sosfiltfilt(sos,mean,axis=1))[0],4))
for k in [5,11,21]:
    print('medfilt',k,round(sc(median_filter(mean,size=(1,k,1)))[0],4))
s,e=sc(mean)
import pandas as pd
print(pd.DataFrame({'a':meta.action,'s':ses,'e':e}).pivot(index='s',columns='a',values='e').round(3))
