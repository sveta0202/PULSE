import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import ACTIONS, load_all, day_folds, score
from feats import build, DS
from nn import Block, prep_input, upsample, onehot, finger_weights


class RetrievalNet(nn.Module):
    def __init__(self, cin, hidden, depth, k, drop, heads):
        super().__init__()
        self.inp = nn.Conv1d(cin + 8, hidden, 1)
        self.blocks = nn.Sequential(*[Block(hidden, k, 2 ** i, drop) for i in range(depth)])
        self.yproj = nn.Linear(20, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.rnn = nn.LSTM(hidden * 2, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.out = nn.Linear(hidden * 2 + 8, 20)

    def encode(self, x, a):
        x = torch.cat([x, a[:, :, None].expand(-1, -1, x.shape[2])], 1)
        return self.blocks(self.inp(x)).transpose(1, 2)

    def forward(self, x, a, mx, my):
        h = self.encode(x, a)
        km = self.encode(mx, a)
        vm = km + self.yproj(my)
        o, _ = self.attn(self.norm(h), self.norm(km), vm, need_weights=False)
        z, _ = self.rnn(torch.cat([h, o], 2))
        return self.out(torch.cat([z, a[:, None, :].expand(-1, z.shape[1], -1)], 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--crop", type=int, default=400)
    p.add_argument("--mem_k", type=int, default=2)
    p.add_argument("--mem_len", type=int, default=600)
    p.add_argument("--gain_aug", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--feat", default="env")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--tag", default="retr")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Path("outputs").mkdir(exist_ok=True)
    t0 = time.time()
    meta, xs, ys = load_all()
    E = {}
    for r in meta.recording_id:
        E[r], mask = prep_input(build(xs[r], a.feat), "none", a.feat)
    Yd = {r: ys[r][::DS] for r in meta.recording_id}
    T = max(len(e) for e in E.values())
    allc = np.zeros((len(meta), 8, T, 20), np.float32)
    for tr, va in day_folds(meta):
        rids = list(tr.recording_id)
        acts = np.array([ACTIONS.index(x) for x in tr.action])
        sess = tr.session_id.values
        mu = np.concatenate([E[r] for r in rids]).mean(0)
        sd = np.concatenate([E[r] for r in rids]).std(0) + 1e-6
        Xt = [torch.tensor((E[r] - mu) / sd, dtype=torch.float32, device=dev) for r in rids]
        Yt = [torch.tensor(Yd[r], dtype=torch.float32, device=dev) for r in rids]
        Wt = torch.tensor(np.stack([finger_weights(x) for x in tr.action]), device=dev)
        At = torch.tensor(np.stack([onehot(x) for x in tr.action]), device=dev)
        gmask = torch.tensor(mask / sd, dtype=torch.float32, device=dev)
        cidx = torch.arange(Xt[0].shape[1], device=dev) % 6
        nets = []
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = RetrievalNet(Xt[0].shape[1], a.hidden, a.depth, a.k, a.drop, a.heads).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)

            def gain(x):
                g = torch.randn(x.shape[0], 1, 6, device=dev) * a.gain_aug + torch.randn(x.shape[0], 1, 1, device=dev) * a.gain_aug
                return x + g[:, :, cidx] * gmask + torch.randn_like(x) * a.noise

            net.train()
            for step in range(a.steps):
                idx = rng.integers(0, len(rids), a.bs)
                xb, yb, mx, my = [], [], [], []
                for i in idx:
                    s = rng.integers(0, len(Xt[i]) - a.crop)
                    xb.append(Xt[i][s:s + a.crop])
                    yb.append(Yt[i][s:s + a.crop])
                    pool = np.where((acts == acts[i]) & (sess != sess[i]))[0]
                    mxs, mys = [], []
                    for j in rng.choice(pool, a.mem_k, replace=len(pool) < a.mem_k):
                        s2 = rng.integers(0, len(Xt[j]) - a.mem_len)
                        mxs.append(Xt[j][s2:s2 + a.mem_len])
                        mys.append(Yt[j][s2:s2 + a.mem_len])
                    mx.append(torch.cat(mxs))
                    my.append(torch.cat(mys))
                xb, yb, mx, my = map(torch.stack, (xb, yb, mx, my))
                pr = net(gain(xb).transpose(1, 2), At[idx], gain(mx).transpose(1, 2), my)
                loss = ((pr - yb).abs() * Wt[idx][:, None, :]).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
            nets.append(net.eval())
        with torch.no_grad():
            for r in va.itertuples():
                x = torch.tensor((E[r.recording_id] - mu) / sd, dtype=torch.float32, device=dev)[None].transpose(1, 2)
                for k, act in enumerate(ACTIONS):
                    pool = np.where(acts == k)[0]
                    mx = torch.cat([Xt[j] for j in pool])[None].transpose(1, 2)
                    my = torch.cat([Yt[j] for j in pool])[None]
                    av = torch.tensor(onehot(act), device=dev)[None]
                    allc[r.Index, k, :x.shape[2]] = np.mean([n(x, av, mx, my)[0].cpu().numpy() for n in nets], 0)
        print("fold", round(time.time() - t0), flush=True)
    np.save(f"outputs/oof_{a.tag}.npy", allc)
    preds = {r.recording_id: upsample(allc[r.Index, ACTIONS.index(r.action), :len(E[r.recording_id])], len(ys[r.recording_id])) for r in meta.itertuples()}
    s, df = score(meta, preds, ys)
    res = {"tag": a.tag, "oracle": round(s, 4), "per_action": df.groupby("action").err.mean().round(3).to_dict(), "args": vars(a), "sec": round(time.time() - t0)}
    print(json.dumps(res), flush=True)
    with open("outputs/nn_results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
