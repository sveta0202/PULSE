import argparse
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.ndimage import uniform_filter1d
from common import ACTIONS, load_all, day_folds, score
from feats import envelopes, lagstack
from nn import upsample


def frame_features(env, span, step):
    lags = lagstack(env, list(range(-span, span + 1, step)))
    stats = []
    for w in (25, 50, 100):
        m = uniform_filter1d(env, w, axis=0, mode="nearest")
        sq = uniform_filter1d(env ** 2, w, axis=0, mode="nearest")
        stats += [m, np.sqrt(np.maximum(sq - m ** 2, 0))]
    glob = np.repeat(np.median(env, 0, keepdims=True), len(env), 0)
    return np.concatenate([lags] + stats + [glob], 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--span", type=int, default=40)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--sub", type=int, default=2)
    p.add_argument("--tag", default="gbm")
    a = p.parse_args()
    Path("outputs").mkdir(exist_ok=True)
    t0 = time.time()
    meta, xs, ys = load_all()
    Fe = {r: frame_features(envelopes(xs[r]), a.span, a.step) for r in meta.recording_id}
    Yd = {r: ys[r][::10] for r in meta.recording_id}
    T = max(len(f) for f in Fe.values())
    allc = np.zeros((len(meta), 8, T, 20), np.float32)
    params = dict(objective="l1", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=50, feature_fraction=0.5,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)
    for tr, va in day_folds(meta):
        for k, act in enumerate(ACTIONS):
            rids = tr[tr.action == act].recording_id
            X = np.concatenate([Fe[r][100::a.sub] for r in rids])
            Y = np.concatenate([Yd[r][100::a.sub] for r in rids])
            for j in range(20):
                m = lgb.train(params, lgb.Dataset(X, Y[:, j]), a.rounds)
                for r in va.itertuples():
                    f = Fe[r.recording_id]
                    allc[r.Index, k, :len(f), j] = m.predict(f)
        print("fold done", round(time.time() - t0), flush=True)
    np.save(f"outputs/oof_{a.tag}.npy", allc)
    preds = {}
    for r in meta.itertuples():
        n = len(Fe[r.recording_id])
        preds[r.recording_id] = upsample(allc[r.Index, ACTIONS.index(r.action), :n], len(ys[r.recording_id]))
    s, df = score(meta, preds, ys)
    res = {"tag": a.tag, "oracle": round(s, 4), "per_action": df.groupby("action").err.mean().round(3).to_dict(), "args": vars(a), "sec": round(time.time() - t0)}
    print(json.dumps(res), flush=True)
    with open("outputs/nn_results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
