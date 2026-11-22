import numpy as np, itertools
exec(open('experiments/ens_round.py').read().split("tags=sys.argv")[0])
fam={'raw':['epl_raw_h96_s1500','epl_raw_h96_s1500_con','epl_raw_h96_s2000','epl_lstmraw_s1500','epl_raw_h96'],
     'tan':['epl_tan_h96','epl_tan','epl_tan_s1500'],
     'corr':['epl_corr_lstm','epl_corr_h96','epl_corr_lstm_s1500'],
     'env':['epl_envd','epl_w1','epl_w2']}
F={k:np.mean([np.load(f'outputs/oof_{t}.npy',mmap_mode='r')[np.arange(80),yl][:, :2051] for t in v],0) for k,v in fam.items()}
for k in F: print(k, round(sc(F[k]),4))
R=[]
for w in itertools.product([0,1,2,3],repeat=4):
    if sum(w)==0: continue
    R.append((sc(sum(wi*F[k] for wi,k in zip(w,F))/sum(w)),w))
R.sort(); [print(dict(zip(F,w)),round(s,4)) for s,w in R[:8]]
