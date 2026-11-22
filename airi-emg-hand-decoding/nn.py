import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import ACTIONS, ACTIVE, CONTEXT, load_all, day_folds, score
from feats import build, DS


def finger_weights(action, active_w=0.5):
    w = np.full(5, 0.2, np.float32)
    a = ACTIVE.get(action)
    if a is not None:
        w[:] = (1 - active_w) / 4
        w[a] = active_w
    return np.repeat(w, 4) / 4 * 20


class Block(nn.Module):
    def __init__(self, c, k, d, drop):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=(k - 1) // 2 * d, dilation=d)
        self.norm = nn.GroupNorm(1, c)
        self.pw = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.pw(F.gelu(self.norm(self.conv(x))))
        return x + self.drop(h)


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class UNet1d(nn.Module):
    def __init__(self, c, levels=3, drop=0.2):
        super().__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        ch = [c * (2 ** min(i, 2)) for i in range(levels + 1)]
        for i in range(levels):
            self.down.append(nn.Sequential(nn.Conv1d(ch[i], ch[i + 1], 5, stride=2, padding=2), nn.GroupNorm(1, ch[i + 1]), nn.GELU(), Block(ch[i + 1], 5, 1, drop)))
        for i in reversed(range(levels)):
            self.up.append(nn.Sequential(nn.Conv1d(ch[i + 1] + ch[i], ch[i], 5, padding=2), nn.GroupNorm(1, ch[i]), nn.GELU(), Block(ch[i], 5, 1, drop)))

    def forward(self, x):
        skips = [x]
        for d in self.down:
            x = d(x)
            skips.append(x)
        skips.pop()
        for u in self.up:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[2], mode="linear", align_corners=False)
            x = u(torch.cat([x, s], 1))
        return x


class Net(nn.Module):
    def __init__(self, cin, hidden=64, n_actions=8, depth=5, k=5, drop=0.2, gru=True, cond=True, pca=None, ctx=0, moe=0):
        super().__init__()
        self.moe = moe
        self.ctx = ctx
        self.pca = pca is not None
        if self.pca:
            self.register_buffer("mu_a", torch.tensor(pca[0], dtype=torch.float32))
            self.register_buffer("basis_a", torch.tensor(pca[1], dtype=torch.float32))
        self.cond = cond
        self.inp = nn.Conv1d(cin + (n_actions if cond else 0), hidden, 1)
        self.blocks = nn.Sequential(*[Block(hidden, k, 2 ** i, drop) for i in range(depth)])
        self.kind = int(gru)
        if self.kind in (4, 5):
            self.blocks = nn.Sequential(Block(hidden, k, 1, drop), UNet1d(hidden, 3, drop))
        if self.kind == 5:
            self.gru = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        elif self.kind == 1:
            self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        elif self.kind == 2:
            self.gru = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        elif self.kind == 3:
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, drop, batch_first=True, norm_first=True)
            self.gru = nn.TransformerEncoder(layer, 2)
        else:
            self.gru = None
        width = hidden * (2 if self.kind in (1, 2, 5) else 1)
        self.out = nn.Linear(width * (2 if ctx else 1) + (n_actions if cond else 0), pca[1].shape[1] if pca is not None else 20)
        if moe:
            self.moe_in = width * (2 if ctx else 1)
            self.moe_w = nn.Parameter(torch.zeros(n_actions, self.moe_in, self.out.out_features))
            self.moe_b = nn.Parameter(torch.zeros(n_actions, self.out.out_features))
        self.n_actions = n_actions

    def encode(self, x, a):
        if self.cond:
            x = torch.cat([x, a[:, :, None].expand(-1, -1, x.shape[2])], 1)
        h = self.blocks(self.inp(x)).transpose(1, 2)
        if self.kind in (1, 2, 5):
            h, _ = self.gru(h)
        elif self.kind == 3:
            h = self.gru(h)
        return h

    def forward(self, x, a):
        if self.cond:
            ao = a[:, :, None].expand(-1, -1, x.shape[2])
            x = torch.cat([x, ao], 1)
        h = self.blocks(self.inp(x)).transpose(1, 2)
        if self.kind in (1, 2, 5):
            h, _ = self.gru(h)
        elif self.kind == 3:
            h = self.gru(h)
        if self.ctx:
            h = torch.cat([h, h.mean(1, keepdim=True).expand(-1, h.shape[1], -1)], 2)
        if self.cond:
            h = torch.cat([h, a[:, None, :].expand(-1, h.shape[1], -1)], 2)
        o = self.out(h)
        if self.moe:
            o = o + torch.einsum("bth,ahj,ba->btj", h[..., :self.moe_in], self.moe_w, a) + (a @ self.moe_b)[:, None, :]
        if self.pca:
            mu = a @ self.mu_a
            basis = torch.einsum("ba,akj->bkj", a, self.basis_a)
            return mu[:, None, :] + torch.einsum("btk,bkj->btj", o, basis)
        return o


def action_pca(Y, A, k):
    act = np.stack(A).argmax(1)
    mus, bases = np.zeros((8, 20), np.float32), np.zeros((8, k, 20), np.float32)
    for c in range(8):
        Yc = np.concatenate([y[100:] for y, ac in zip(Y, act) if ac == c])
        mus[c] = np.median(Yc, 0)
        _, _, vt = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False)
        bases[c] = vt[:k]
    return mus, bases


def add_mvc(E, mask, meta, mode=1, sessions=None, actions=None):
    sessions = meta.session_id.values if sessions is None else sessions
    actions = meta.action.values if actions is None else actions
    rids = list(meta.recording_id)
    out = {}
    for s in np.unique(sessions):
        idx = [i for i in range(len(rids)) if sessions[i] == s]
        allf = np.concatenate([E[rids[i]][:, :18] for i in idx])
        ref = {actions[i]: np.median(E[rids[i]][:, :18], 0) for i in idx}
        for i in idx:
            e = E[rids[i]]
            if mode == 1:
                extra = e[:, :18] - np.median(allf, 0)
            else:
                extra = np.concatenate([e[:, :18] - ref["Open"], e[:, :18] - ref["Close"]], 1)
            out[rids[i]] = np.concatenate([e, extra], 1)
    return out, np.r_[mask, np.zeros(18 if mode == 1 else 36, np.float32)]


def log_mask(kind):
    if kind == "env":
        return np.ones(18, np.float32)
    if kind == "frame":
        return np.r_[np.ones(12), np.zeros(18)].astype(np.float32)
    if kind == "envraw":
        return np.r_[np.ones(18), np.zeros(60)].astype(np.float32)
    if kind == "envcorr":
        return np.r_[np.ones(18), np.zeros(45)].astype(np.float32)
    if kind == "envtan":
        return np.r_[np.ones(18), np.zeros(63)].astype(np.float32)
    if kind == "allfeat":
        return np.r_[np.ones(18), np.zeros(45 + 63 + 60)].astype(np.float32)
    if kind == "envd":
        return np.r_[np.ones(18), np.zeros(18)].astype(np.float32)
    return np.r_[np.ones(30), np.zeros(18)].astype(np.float32)


def prep_input(env, mode, kind="env"):
    m = log_mask(kind)
    if mode == "center":
        return env - np.median(env, 0) * m, m
    if mode == "both":
        return np.concatenate([env, env - np.median(env, 0)], 1), np.r_[m, np.zeros_like(m)]
    return env, m


def action_stats(X, actions):
    st = {}
    for act in set(actions):
        xs_ = [x for x, a_ in zip(X, actions) if a_ == act]
        st[act] = (np.mean([np.median(x, 0) for x in xs_], 0), np.mean([x.std(0) for x in xs_], 0))
    return st


def align(e, act, stats, mask, mode):
    mu, sd = stats[act]
    m = mask.astype(bool)
    out = e.copy()
    med = np.median(e, 0)
    if mode == 1:
        out[:, m] = e[:, m] - med[m] + mu[m]
    else:
        out[:, m] = (e[:, m] - med[m]) / (e[:, m].std(0) + 1e-6) * sd[m] + mu[m]
    return out


def upsample(p, T):
    t = np.arange(T) / DS
    i0 = np.clip(np.floor(t).astype(int), 0, len(p) - 1)
    i1 = np.clip(i0 + 1, 0, len(p) - 1)
    w = (t - np.floor(t))[:, None]
    return p[i0] * (1 - w) + p[i1] * w


class Trainer:
    def __init__(self, args, device):
        self.a = args
        self.dev = device

    def pretrain(self, net, Xall, rng):
        a = self.a
        width = net.out.in_features - (8 if a.cond else 0)
        head = nn.Linear(width, Xall[0].shape[1]).to(self.dev)
        opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=a.wd)
        L = a.crop
        for step in range(a.ssl_steps):
            idx = rng.integers(0, len(Xall), a.bs)
            xb = torch.stack([Xall[i][(s := rng.integers(0, len(Xall[i]) - L)):s + L] for i in idx])
            m = torch.ones(a.bs, L, 1, device=self.dev)
            for b in range(a.bs):
                for _ in range(int(L * 0.3 / 10)):
                    t0 = rng.integers(0, L - 10)
                    m[b, t0:t0 + 10] = 0
            xin = xb * m
            h = net.encode(xin.transpose(1, 2), torch.zeros(a.bs, 8, device=self.dev))
            rec = head(h)
            loss = ((rec - xb).abs() * (1 - m)).sum() / ((1 - m).sum() * xb.shape[2] + 1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def fit(self, X, Y, A, W, seed, mask, Xu=None):
        a = self.a
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        mu = np.concatenate(X).mean(0)
        sd = np.concatenate(X).std(0) + 1e-6
        self.mu, self.sd = mu, sd
        self.mask_ = mask
        self.pca = action_pca(Y, A, a.pca_k) if a.pca_k > 0 else None
        net = Net(X[0].shape[1], a.hidden, depth=a.depth, k=a.k, drop=a.drop, gru=a.gru, cond=a.cond, pca=self.pca, ctx=a.ctx, moe=a.moe).to(self.dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = a.steps
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
        Xt = [torch.tensor((x - mu) / sd, dtype=torch.float32, device=self.dev) for x in X]
        Yt = [torch.tensor(y, dtype=torch.float32, device=self.dev) for y in Y]
        At = torch.tensor(np.stack(A), dtype=torch.float32, device=self.dev)
        Wt = torch.tensor(np.stack(W), dtype=torch.float32, device=self.dev)
        L = a.crop
        n = len(X)
        dann_head = None
        if a.dann > 0 and getattr(self, "sessions", None) is not None:
            sess_codes = {s_: i for i, s_ in enumerate(sorted(set(self.sessions)))}
            St = torch.tensor([sess_codes[s_] for s_ in self.sessions], device=self.dev)
            width = net.out.in_features - (8 if a.cond else 0)
            dann_head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, len(sess_codes))).to(self.dev)
            dopt = torch.optim.Adam(dann_head.parameters(), lr=1e-3)
        if a.ssl_steps > 0:
            Xall = Xt + [torch.tensor((x - mu) / sd, dtype=torch.float32, device=self.dev) for x in (Xu or []) if len(x) > L + 1]
            self.pretrain(net, Xall, rng)
        nch = X[0].shape[1]
        ngroups = nch // 6
        gmask = torch.tensor(mask / sd, dtype=torch.float32, device=self.dev)
        chan_idx = torch.arange(nch, device=self.dev) % 6
        ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema)) if a.ema > 0 else None
        net.train()
        act_ids = np.stack(A).argmax(1)
        by_act = {k: np.where(act_ids == k)[0] for k in np.unique(act_ids)}

        def crops(ids):
            xb, yb = [], []
            for i in ids:
                T = len(Xt[i])
                f = float(np.exp(rng.uniform(-a.stretch, a.stretch))) if a.stretch > 0 else 1.0
                Ls = min(int(round(L * f)), T - 1)
                s = rng.integers(0, T - Ls)
                xc, yc = Xt[i][s:s + Ls], Yt[i][s:s + Ls]
                if Ls != L:
                    xc = F.interpolate(xc.T[None], size=L, mode="linear", align_corners=True)[0].T
                    yc = F.interpolate(yc.T[None], size=L, mode="linear", align_corners=True)[0].T
                xb.append(xc)
                yb.append(yc)
            return torch.stack(xb), torch.stack(yb)

        for step in range(steps):
            idx = rng.integers(0, n, a.bs)
            xb, yb = crops(idx)
            if a.mixup > 0:
                partner = np.array([rng.choice(by_act[act_ids[i]]) for i in idx])
                xb2, yb2 = crops(partner)
                lam = torch.tensor(rng.beta(a.mixup, a.mixup, a.bs), dtype=torch.float32, device=self.dev)[:, None, None]
                xb = lam * xb + (1 - lam) * xb2
                yb = lam * yb + (1 - lam) * yb2
            if a.chdrop > 0:
                keep = (torch.rand(a.bs, 1, 6, device=self.dev) > a.chdrop).float()[:, :, chan_idx]
                xb = xb * keep
            if a.mixch > 0:
                Mx = torch.eye(6, device=self.dev)[None] + a.mixch * torch.randn(a.bs, 6, 6, device=self.dev)
                Mx = Mx / Mx.abs().sum(2, keepdim=True)
                lin = torch.exp(xb[:, :, :18].reshape(a.bs, L, 3, 6) * torch.tensor(sd[:18], device=self.dev).reshape(1, 1, 3, 6) + torch.tensor(mu[:18], device=self.dev).reshape(1, 1, 3, 6))
                lin = torch.einsum("btcj,bkj->btck", lin, Mx.abs())
                xb = torch.cat([((torch.log(lin + 1e-6).reshape(a.bs, L, 18) - torch.tensor(mu[:18], device=self.dev)) / torch.tensor(sd[:18], device=self.dev)), xb[:, :, 18:]], 2)
            if a.flip > 0:
                fl = torch.rand(a.bs, device=self.dev) < a.flip
                xb = torch.where(fl[:, None, None], xb.flip(1), xb)
                yb = torch.where(fl[:, None, None], yb.flip(1), yb)
            if a.contrast > 0:
                c = torch.exp(torch.randn(a.bs, 1, 6, device=self.dev) * a.contrast)[:, :, chan_idx]
                m = torch.tensor(mask, device=self.dev, dtype=torch.float32).bool()
                mean = xb.mean(1, keepdim=True)
                xb = torch.where(m, mean + (xb - mean) * c, xb)
            if a.gain_aug > 0:
                g = torch.randn(a.bs, 1, 6, device=self.dev) * a.gain_aug + torch.randn(a.bs, 1, 1, device=self.dev) * a.gain_aug
                xb = xb + g[:, :, chan_idx] * gmask
            if a.noise > 0:
                xb = xb + torch.randn_like(xb) * a.noise
            ab = At[idx]
            if a.label_noise > 0:
                ab = ab * (1 - a.label_noise) + a.label_noise / 8
            if a.cfg_drop > 0:
                ab = ab * (torch.rand(a.bs, 1, device=self.dev) > a.cfg_drop).float()
            if dann_head is not None:
                hfeat = net.encode(xb.transpose(1, 2), ab)
                p = net.out(torch.cat([hfeat, ab[:, None, :].expand(-1, hfeat.shape[1], -1)], 2))
                lam = a.dann * min(1.0, step / (0.3 * steps))
                dl = F.cross_entropy(dann_head(GradRev.apply(hfeat.mean(1), lam)), St[idx])
            else:
                p = net(xb.transpose(1, 2), ab)
            loss = ((p - yb).abs() * Wt[idx][:, None, :]).mean()
            if dann_head is not None:
                loss = loss + dl
            if a.vel_w > 0:
                loss = loss + a.vel_w * (((p[:, 1:] - p[:, :-1]) - (yb[:, 1:] - yb[:, :-1])).abs() * Wt[idx][:, None, :]).mean() * 10
            opt.zero_grad()
            if dann_head is not None:
                dopt.zero_grad()
            loss.backward()
            if a.sam > 0:
                params = [q for q in net.parameters() if q.grad is not None]
                gn = torch.norm(torch.stack([q.grad.norm() for q in params])) + 1e-12
                eps = [a.sam * q.grad / gn for q in params]
                with torch.no_grad():
                    for q, e in zip(params, eps):
                        q.add_(e)
                opt.zero_grad()
                p2 = net(xb.transpose(1, 2), ab)
                ((p2 - yb).abs() * Wt[idx][:, None, :]).mean().backward()
                with torch.no_grad():
                    for q, e in zip(params, eps):
                        q.sub_(e)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if dann_head is not None:
                dopt.step()
            if ema is not None:
                ema.update_parameters(net)
        self.net = (ema.module if ema is not None else net).eval()
        return self

    @torch.no_grad()
    def predict(self, x, a_vec):
        a = self.a
        xt = torch.tensor((x - self.mu) / self.sd, dtype=torch.float32, device=self.dev)[None].transpose(1, 2)
        at = torch.tensor(a_vec, dtype=torch.float32, device=self.dev)[None]
        out = self.net(xt, at)
        if getattr(a, "tta", 0) > 0:
            g = torch.Generator(device="cpu").manual_seed(0)
            mask = torch.tensor(self.mask_ / self.sd, dtype=torch.float32, device=self.dev)
            nch = x.shape[1]
            idx = torch.arange(nch) % 6
            outs = [out]
            for _ in range(a.tta):
                d = (torch.randn(6, generator=g) * a.gain_aug * 0.5 + torch.randn(1, generator=g) * a.gain_aug * 0.5).to(self.dev)
                outs.append(self.net(xt + (d[idx] * mask)[None, :, None], at))
            out = torch.stack(outs).median(0).values
        if getattr(a, "cfg_w", 0) != 0:
            un = self.net(xt, torch.zeros_like(at))
            out = out + a.cfg_w * (out - un)
        return out[0].cpu().numpy()


def onehot(action):
    v = np.zeros(8, np.float32)
    v[ACTIONS.index(action)] = 1
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--gru", type=int, default=1)
    p.add_argument("--cond", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--crop", type=int, default=400)
    p.add_argument("--gain_aug", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--label_noise", type=float, default=0.0)
    p.add_argument("--inmode", default="none")
    p.add_argument("--feat", default="env")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--stretch", type=float, default=0.0)
    p.add_argument("--align", type=int, default=0)
    p.add_argument("--pseudo", type=int, default=0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--pca_k", type=int, default=0)
    p.add_argument("--ctx", type=int, default=0)
    p.add_argument("--mvc", type=int, default=0)
    p.add_argument("--mixch", type=float, default=0.0)
    p.add_argument("--flip", type=float, default=0.0)
    p.add_argument("--cfg_drop", type=float, default=0.0)
    p.add_argument("--cfg_w", type=float, default=0.0)
    p.add_argument("--tta", type=int, default=0)
    p.add_argument("--active_w", type=float, default=0.5)
    p.add_argument("--dann", type=float, default=0.0)
    p.add_argument("--moe", type=int, default=0)
    p.add_argument("--pseudo_src", default="")
    p.add_argument("--contrast", type=float, default=0.0)
    p.add_argument("--sam", type=float, default=0.0)
    p.add_argument("--ssl_steps", type=int, default=0)
    p.add_argument("--vel_w", type=float, default=0.0)
    p.add_argument("--pseudo_w", type=float, default=1.0)
    p.add_argument("--chdrop", type=float, default=0.0)
    p.add_argument("--probs", default="")
    p.add_argument("--tag", default="run")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Path("outputs").mkdir(exist_ok=True)
    t0 = time.time()
    meta, xs, ys = load_all()
    E = {}
    for r in meta.recording_id:
        E[r], mask = prep_input(build(xs[r], a.feat), a.inmode, a.feat)
    if a.mvc:
        E, mask = add_mvc(E, mask, meta, a.mvc)
    Yd = {r: ys[r][::DS] for r in meta.recording_id}
    probs = np.load(a.probs) if a.probs else None
    preds, preds_hat = {}, {}
    psrc = np.load(a.pseudo_src, mmap_mode="r") if a.pseudo_src else None
    allc = np.zeros((len(meta), 8, max(len(e) for e in E.values()), 20), np.float32)
    for tr, va in day_folds(meta):
        stats = action_stats([E[r] for r in tr.recording_id], list(tr.action)) if a.align else None
        al = lambda e, act: align(e, act, stats, mask, a.align) if a.align else e
        X = [al(E[r], act) for r, act in zip(tr.recording_id, tr.action)]
        Y = [Yd[r] for r in tr.recording_id]
        A = [onehot(x) for x in tr.action]
        W = [finger_weights(x, a.active_w) for x in tr.action]
        Xu = [al(E[r.recording_id], r.action) for r in va.itertuples()]
        models = []
        for s in range(a.seeds):
            t = Trainer(a, dev)
            t.sessions = list(tr.session_id)
            models.append(t.fit(X, Y, A, W, s, mask, Xu))
        for rnd in range(a.pseudo):
            Xv = [al(E[r.recording_id], r.action) for r in va.itertuples()]
            if a.pseudo_src:
                Yv = [psrc[r.Index, ACTIONS.index(r.action), :len(x)] for x, r in zip(Xv, va.itertuples())]
            else:
                Yv = [np.mean([m.predict(x, onehot(r.action)) for m in models], 0) for x, r in zip(Xv, va.itertuples())]
            Av = [onehot(x) for x in va.action]
            Wv = [finger_weights(x) * a.pseudo_w for x in va.action]
            models = [Trainer(a, dev).fit(X + Xv, Y + Yv, A + Av, W + Wv, 50 + s + 10 * rnd, mask) for s in range(a.seeds)]
        for r in va.itertuples():
            e = E[r.recording_id]
            for k, act in enumerate(ACTIONS):
                allc[r.Index, k, :len(e)] = np.mean([m.predict(al(e, act), onehot(act)) for m in models], 0)
            T = len(ys[r.recording_id])
            preds[r.recording_id] = upsample(allc[r.Index, ACTIONS.index(r.action), :len(e)], T)
            if probs is not None:
                k = int(probs[r.Index].argmax())
                preds_hat[r.recording_id] = upsample(allc[r.Index, k, :len(e)], T)
    np.save(f"outputs/oof_{a.tag}.npy", allc)
    s, df = score(meta, preds, ys)
    res = {"tag": a.tag, "oracle": round(s, 4), "per_action": df.groupby("action").err.mean().round(3).to_dict(), "args": vars(a), "sec": round(time.time() - t0)}
    if probs is not None:
        s2, _ = score(meta, preds_hat, ys)
        res["pred_action"] = round(s2, 4)
    print(json.dumps(res), flush=True)
    with open("outputs/nn_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
