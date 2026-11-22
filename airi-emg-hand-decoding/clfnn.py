import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import ACTIONS, load_all, day_folds
from feats import build
from nn import Block, prep_input


class ClfNet(nn.Module):
    def __init__(self, cin, hidden, depth, k, drop):
        super().__init__()
        self.inp = nn.Conv1d(cin, hidden, 1)
        self.blocks = nn.Sequential(*[Block(hidden, k, 2 ** i, drop) for i in range(depth)])
        self.head = nn.Linear(hidden * 2, 8)

    def forward(self, x):
        h = self.blocks(self.inp(x))
        z = torch.cat([h.mean(2), h.std(2)], 1)
        return self.head(z)


def fit_predict(Xtr, ytr, Xte, a, dev, seed, mask):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu = np.concatenate(Xtr).mean(0)
    sd = np.concatenate(Xtr).std(0) + 1e-6
    Xt = [torch.tensor((x - mu) / sd, dtype=torch.float32, device=dev) for x in Xtr]
    yt = torch.tensor(ytr, device=dev)
    net = ClfNet(Xtr[0].shape[1], a.hidden, a.depth, a.k, a.drop).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    nch = Xtr[0].shape[1]
    gmask = torch.tensor(mask / sd, device=dev, dtype=torch.float32)
    net.train()
    for step in range(a.steps):
        idx = rng.integers(0, len(Xt), a.bs)
        L = a.crop
        xb = torch.stack([Xt[i][(s := rng.integers(0, len(Xt[i]) - L)):s + L] for i in idx])
        g = torch.randn(a.bs, 1, 6, device=dev) * a.gain_aug + torch.randn(a.bs, 1, 1, device=dev) * a.gain_aug
        xb = xb + g[:, :, torch.arange(nch, device=dev) % 6] * gmask + torch.randn_like(xb) * a.noise
        loss = F.cross_entropy(net(xb.transpose(1, 2)), yt[idx], label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    out = []
    with torch.no_grad():
        for x in Xte:
            xt = torch.tensor((x - mu) / sd, dtype=torch.float32, device=dev)
            L = min(a.crop, len(xt))
            starts = np.arange(0, len(xt) - L + 1, max(L // 4, 1))
            xb = torch.stack([xt[s:s + L] for s in starts]).transpose(1, 2)
            lp = F.log_softmax(net(xb), 1).mean(0)
            out.append(lp.cpu().numpy())
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--drop", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=5e-2)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--crop", type=int, default=500)
    p.add_argument("--gain_aug", type=float, default=0.3)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--inmode", default="none")
    p.add_argument("--feat", default="env")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--tag", default="clf")
    p.add_argument("--selftrain", type=int, default=0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Path("outputs").mkdir(exist_ok=True)
    meta, xs, ys = load_all()
    E = {}
    for r in meta.recording_id:
        E[r], mask = prep_input(build(xs[r], a.feat), a.inmode, a.feat)
    yl = meta.action.map({x: i for i, x in enumerate(ACTIONS)}).values
    LP = np.zeros((len(meta), 8))
    for tr, va in day_folds(meta):
        Xtr = [E[r] for r in tr.recording_id]
        Xte = [E[r] for r in va.recording_id]
        lp = np.mean([fit_predict(Xtr, yl[tr.index], Xte, a, dev, s, mask) for s in range(a.seeds)], 0)
        nses = va.session_id.nunique()
        for rnd in range(a.selftrain):
            ri, ci = linear_sum_assignment(np.repeat(-lp, nses, 1))
            pl = np.zeros(len(va), int)
            pl[ri] = ci // nses
            ytr2 = np.concatenate([yl[tr.index], pl])
            lp = np.mean([fit_predict(Xtr + Xte, ytr2, Xte, a, dev, 100 + s + 10 * rnd, mask) for s in range(a.seeds)], 0)
        LP[va.index] = lp
    P = np.exp(LP)
    P /= P.sum(1, keepdims=True)
    acc = float((P.argmax(1) == yl).mean())
    ll = float(-np.log(P[np.arange(len(P)), yl]).mean())
    np.save(f"outputs/clfnn_{a.tag}.npy", P)
    res = {"tag": a.tag, "acc": acc, "logloss": round(ll, 4), "args": vars(a)}
    print(json.dumps(res), flush=True)
    with open("outputs/clf_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
