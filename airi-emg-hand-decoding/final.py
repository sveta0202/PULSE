import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import ACTIONS, CONTEXT, load_all
from feats import build
from nn import Net, Trainer, action_pca, prep_input, upsample, onehot, finger_weights
from clfnn import fit_predict
from combine import assign
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MODEL_DIR = "outputs/models"
ANGLE_COLS = [f"angle_{i:02d}" for i in range(20)]
ID_COLS = ["row_id", "recording_id", "sample_index"]
REG_DEFAULT = dict(hidden=64, depth=5, k=5, drop=0.2, gru=1, cond=1, lr=2e-3, wd=1e-2, steps=1000, bs=32, crop=400,
                   gain_aug=0.5, noise=0.05, label_noise=0.0, inmode="none", feat="env", ema=0.0, stretch=0.0, chdrop=0.0,
                   pseudo=0, pseudo_w=0.5, seeds=2, mixup=0.0, vel_w=0.0, ssl_steps=0, pca_k=0, ctx=0, moe=0, active_w=0.5, dann=0.0, mixch=0.0, flip=0.0, cfg_drop=0.0, cfg_w=0.0, tta=0, contrast=0.0, pseudo_src="", sam=0.0)
CLF_DEFAULT = dict(hidden=64, depth=6, k=5, drop=0.3, lr=2e-3, wd=5e-2, steps=800, bs=64, crop=500, gain_aug=0.3, noise=0.05,
                   inmode="none", feat="env", seeds=3)


def classify(meta, xs, xt, rids, clf_cfgs, dev, use_lr):
    yl = meta.action.map({x: i for i, x in enumerate(ACTIONS)}).values
    logps = []
    for cfg in clf_cfgs:
        c = argparse.Namespace(**{**CLF_DEFAULT, **cfg})
        Etr, Ete = {}, {}
        for r in meta.recording_id:
            Etr[r], mask = prep_input(build(xs[r], c.feat), c.inmode, c.feat)
        for r in rids:
            Ete[r], _ = prep_input(build(xt[r], c.feat), c.inmode, c.feat)
        Xtr = [Etr[r] for r in meta.recording_id]
        logps.append(np.mean([fit_predict(Xtr, yl, [Ete[r] for r in rids], c, dev, s, mask) for s in range(c.seeds)], 0))
        print("clf done", cfg, flush=True)
    if use_lr:
        from clf_features import rec_feats
        Ftr = np.stack([rec_feats(xs[r]) for r in meta.recording_id])
        Fte = np.stack([rec_feats(xt[r]) for r in rids])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(Ftr, yl)
        logps.append(np.log(m.predict_proba(Fte) + 1e-9))
    P = np.exp(np.mean(logps, 0))
    return P / P.sum(1, keepdims=True)


def cached_fit(c, dev, X, Y, A, W, seed, mask, Xu=None):
    if c.ssl_steps > 0:
        return Trainer(c, dev).fit(X, Y, A, W, seed, mask, Xu)
    late = dict(pca_k=0, ctx=0, moe=0, active_w=0.5, dann=0.0, mixch=0.0, flip=0.0, cfg_drop=0.0, cfg_w=0.0, tta=0, contrast=0.0, pseudo_src="", sam=0.0)
    key = hashlib.md5(json.dumps({k: v for k, v in sorted(vars(c).items()) if k not in ("seeds", "pseudo", "pseudo_w") and not (k in late and v == late[k])}).encode()).hexdigest()[:12]
    path = Path(MODEL_DIR) / f"{key}_s{seed}.pt"
    t = Trainer(c, dev)
    if path.exists():
        ck = torch.load(path, map_location=dev, weights_only=False)
        t.mu, t.sd = ck["mu"], ck["sd"]
        t.net = Net(X[0].shape[1], c.hidden, depth=c.depth, k=c.k, drop=c.drop, gru=c.gru, cond=c.cond, pca=action_pca(Y, A, c.pca_k) if c.pca_k > 0 else None, ctx=c.ctx, moe=c.moe).to(dev)
        t.net.load_state_dict(ck["state"])
        t.net.eval()
        return t
    t.fit(X, Y, A, W, seed, mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": t.net.state_dict(), "mu": t.mu, "sd": t.sd, "cfg": vars(c)}, path)
    return t


def regress(meta, xs, ys, xt, rids, acts, cfg, dev, teacher=None):
    c = argparse.Namespace(**{**REG_DEFAULT, **{k: v for k, v in cfg.items() if k != "w"}})
    Etr, Ete = {}, {}
    for r in meta.recording_id:
        Etr[r], mask = prep_input(build(xs[r], c.feat), c.inmode, c.feat)
    for r in rids:
        Ete[r], _ = prep_input(build(xt[r], c.feat), c.inmode, c.feat)
    X = [Etr[r] for r in meta.recording_id]
    Y = [ys[r][::10] for r in meta.recording_id]
    A = [onehot(x) for x in meta.action]
    W = [finger_weights(x) for x in meta.action]
    Xv = [Ete[r] for r in rids]
    models = [] if teacher is not None else [cached_fit(c, dev, X, Y, A, W, s, mask, Xv) for s in range(c.seeds)]
    for rnd in range(c.pseudo):
        if teacher is not None:
            Yv = [teacher[i][:len(x)] for i, x in enumerate(Xv)]
        else:
            Yv = [np.mean([m.predict(x, onehot(act)) for m in models], 0) for x, act in zip(Xv, acts)]
        Av = [onehot(x) for x in acts]
        Wv = [finger_weights(x) * c.pseudo_w for x in acts]
        models = [Trainer(c, dev).fit(X + Xv, Y + Yv, A + Av, W + Wv, 50 + s + 10 * rnd, mask) for s in range(c.seeds)]
    print("reg done", cfg, flush=True)
    return [np.mean([m.predict(x, onehot(act)) for m in models], 0) for x, act in zip(Xv, acts)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--use-lr", type=int, default=1)
    p.add_argument("--model-dir", default="outputs/models")
    p.add_argument("--holdout-days", default="")
    p.add_argument("--assign", choices=["balanced", "cap", "argmax"], default="balanced")
    a = p.parse_args()
    global MODEL_DIR
    MODEL_DIR = a.model_dir
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta, xs, ys = load_all()
    if a.holdout_days:
        hold = meta.day_id.isin(a.holdout_days.split(","))
        meta = meta[~hold].reset_index(drop=True)
    template = pd.read_csv(a.template)
    assert template.columns.tolist() == ID_COLS + ANGLE_COLS
    rids = list(dict.fromkeys(template.recording_id))
    xt = {r: np.load(Path(a.test_dir) / f"{r}.npy") for r in rids}
    cfg = json.loads(Path(a.config).read_text())
    P = classify(meta, xs, xt, rids, cfg["clf"], dev, a.use_lr)
    if a.assign == "argmax" or len(rids) % 8 != 0:
        choice = P.argmax(1)
    elif a.assign == "cap":
        choice = assign(P, len(rids) // 8 + 1)
    else:
        choice = assign(P, len(rids) // 8)
    print("argmax vs assigned disagreements:", int((choice != P.argmax(1)).sum()), "counts:", np.bincount(choice, minlength=8).tolist(), "argmax counts:", np.bincount(P.argmax(1), minlength=8).tolist(), flush=True)
    acts = [ACTIONS[k] for k in choice]
    preds = [regress(meta, xs, ys, xt, rids, acts, c, dev) for c in cfg["reg"]]
    ws = np.array([c.get("w", 1.0) for c in cfg["reg"]])
    blend = [np.tensordot(ws / ws.sum(), np.stack([pp[i] for pp in preds]), axes=1) for i in range(len(rids))]
    for rnd, stage in enumerate(cfg.get("students", [])):
        sp = [regress(meta, xs, ys, xt, rids, acts, {**c, "pseudo": 1}, dev, teacher=blend) for c in stage]
        sw = np.array([c.get("w", 1.0) for c in stage])
        blend = [np.tensordot(sw / sw.sum(), np.stack([pp[i] for pp in sp]), axes=1) for i in range(len(rids))]
        preds = preds + sp
        print("student stage done", rnd, flush=True)
    out = template.copy()
    info = []
    for i, r in enumerate(rids):
        pr = blend[i]
        full = np.clip(upsample(pr, len(xt[r])), -np.pi, np.pi)
        rows = (template.recording_id == r).to_numpy()
        idx = template.loc[rows, "sample_index"].to_numpy()
        assert idx.min() >= CONTEXT and idx.max() < len(full)
        out.loc[rows, ANGLE_COLS] = full[idx]
        info.append({"recording_id": r, "action": acts[i], "p_assigned": float(P[i, choice[i]]), "argmax": ACTIONS[int(P[i].argmax())], "p_max": float(P[i].max())})
    vals = out[ANGLE_COLS].to_numpy(np.float64)
    assert np.isfinite(vals).all() and out[ID_COLS].equals(template[ID_COLS])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6f")
    pd.DataFrame(info).to_csv(Path(a.out).with_suffix(".actions.csv"), index=False)
    np.save(Path(a.out).with_suffix(".probs.npy"), P)
    np.save(Path(a.out).with_suffix(".preds.npy"), np.array(preds, dtype=object), allow_pickle=True)
    print(pd.DataFrame(info).to_string())
    print("saved", a.out, out.shape)


if __name__ == "__main__":
    main()
