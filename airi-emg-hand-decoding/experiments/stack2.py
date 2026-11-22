import numpy as np, torch, torch.nn as nn, sys
exec(open('experiments/greedy.py').read().split("for k,v in sorted")[0])
from nn import finger_weights, onehot
sel=['r8_lstm','r10_raw_chd3','r7_envcorr_pseudo','r8_lstm_raw','r4_chd','r5_pseudo1_w05']
B=np.mean([O[k] for k in sel],0)[:, :2050]
Y=np.stack([y[:2050] for y in Ys]); A=np.stack([onehot(a) for a in meta.action]); W=np.stack([finger_weights(a) for a in meta.action])
print('base',round(sc(B),4))
class M(nn.Module):
    def __init__(s,h=32):
        super().__init__(); s.inp=nn.Conv1d(28,h,5,padding=2); s.gru=nn.GRU(h,h,batch_first=True,bidirectional=True); s.out=nn.Linear(2*h,20)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s,x,a):
        z=torch.cat([x,a[:,None,:].expand(-1,x.shape[1],-1)],2)
        h=torch.relu(s.inp(z.transpose(1,2))).transpose(1,2); h,_=s.gru(h); return x+s.out(h)
torch.manual_seed(0)
P=B.copy()
day=meta.day_id.values
for d in np.unique(day):
    tr=np.where(day!=d)[0]; te=np.where(day==d)[0]
    m=M(); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-2)
    Xt=torch.tensor(B,dtype=torch.float32); Yt=torch.tensor(Y,dtype=torch.float32); At=torch.tensor(A); Wt=torch.tensor(W)
    for step in range(int(sys.argv[1]) if len(sys.argv)>1 else 300):
        idx=np.random.choice(tr,16); s=np.random.randint(100,2050-400)
        p=m(Xt[idx,s:s+400],At[idx]); loss=((p-Yt[idx,s:s+400]).abs()*Wt[idx][:,None]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): P[te]=m(Xt[te],At[te]).numpy()
print('stacked',round(sc(P),4))
