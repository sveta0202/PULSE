import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import ACTIONS, load_all, record_error
from nn import upsample


def assign(P, per_action):
    ri, ci = linear_sum_assignment(np.repeat(-np.log(P + 1e-9), per_action, 1))
    out = np.zeros(len(P), int)
    out[ri] = ci // per_action
    return out


def rec_errors(meta, ys, oof, choice):
    errs = np.zeros(len(meta))
    for i, r in enumerate(meta.itertuples()):
        y = ys[r.recording_id]
        p = upsample(oof[i, choice[i], : (len(y) + 9) // 10], len(y))
        errs[i] = record_error(y, p, r.action)
    return errs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof", nargs="+", required=True)
    p.add_argument("--probs", nargs="+", required=True)
    p.add_argument("--draws", type=int, default=300)
    a = p.parse_args()
    meta, xs, ys = load_all()
    oof = np.mean([np.load(f) for f in a.oof], 0)
    P = np.exp(np.mean([np.log(np.load(f) + 1e-9) for f in a.probs], 0))
    P /= P.sum(1, keepdims=True)
    yl = meta.action.map({x: i for i, x in enumerate(ACTIONS)}).values
    E = np.stack([rec_errors(meta, ys, oof, np.full(len(meta), k)) for k in range(8)], 1)
    ses = meta.session_id.values
    us = np.unique(ses)

    def sess_score(err):
        return np.mean([err[ses == s].mean() for s in us])

    print("oracle", round(sess_score(E[np.arange(80), yl]), 4))
    print("argmax", round(sess_score(E[np.arange(80), P.argmax(1)]), 4), "acc", (P.argmax(1) == yl).mean())
    rng = np.random.default_rng(0)
    sc, acc = [], []
    for _ in range(a.draws):
        pick = rng.choice(us, 4, replace=False)
        idx = np.where(np.isin(ses, pick))[0]
        ch = assign(P[idx], 4)
        acc.append((ch == yl[idx]).mean())
        e = E[idx, ch]
        sc.append(np.mean([e[ses[idx] == s].mean() for s in pick]))
    print("balanced(4 sessions)", round(float(np.mean(sc)), 4), "+-", round(float(np.std(sc)), 4), "acc", round(float(np.mean(acc)), 4))


if __name__ == "__main__":
    main()
