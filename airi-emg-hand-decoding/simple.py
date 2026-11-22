"""Упрощённое решение: ЭМГ (6 каналов, 500 Гц) -> 20 углов кисти. Можно запускать по частям.

Весь пайплайн в одном файле, без 30 экспериментальных флагов из nn.py.

  1. Признаки (feats.py): огибающие + сырой сигнал / tangent space / корреляции, 50 Гц.
  2. Классификатор действия: CNN на 3 видах признаков + LogReg -> венгерский алгоритм
     (в тесте каждое действие встречается одинаковое число раз).
  3. Учитель: 3 регрессора TCN + BiLSTM (по одному на вид признаков), метка действия на входе.
  4. Студенты: те же 3 регрессора, обучены на трейне + тесте с метками учителя
     (транздуктивная дистилляция). Ответ — среднее студентов, 50 -> 500 Гц.

Работа по частям. Пайплайн разбит на шаги: 4 шага классификатора, 3 шага учителя, 3 шага студентов
(в CV — так в каждом из 7 фолдов). Каждый шаг сохраняется в папку --work (по умолчанию work/).
Повторный запуск той же команды пропускает готовые шаги и продолжает с места остановки.
--time-limit N — не начинать новый шаг после N минут (чтобы успеть до конца сессии платформы).

Рядом с этим файлом должны лежать common.py, feats.py, clf_features.py.
Папка с данными (train/metadata.csv, train/signals, train/targets) ищется автоматически, иначе --data.

Сколько это стоит. Полное обучение — 27 сетей на фолд; на GPU это минуты, на CPU — часы на каждую сеть.
Поэтому на компьютере без GPU используйте --quick и --fast, а полный cv и predict запускайте на GPU.

Запуск (в ноутбуке — то же с ! в начале или simple.main(["cv", "--fold", "0"])):
  python simple.py cv --quick                  # проверка кода: 1 фолд, 20 шагов (несколько минут на CPU)
  python simple.py cv --fast                   # быстрая оценка на CPU: 1 фолд, 1 сид, шагов /4 (~30-60 мин);
                                               #   числа хуже финальных — сравнивайте варианты между собой
  python simple.py cv --fast --fold 2 3        # то же на других фолдах
  python simple.py cv --fold 0                 # полный режим, один фолд из 7 (0..6); затем --fold 1 и т.д.
  python simple.py cv --time-limit 50          # все фолды, но остановиться через 50 мин; повторять до конца
  python simple.py report                      # итог CV по готовым фолдам (с теми же --fast/--quick/--seeds)
  python simple.py predict --time-limit 50     # тест -> CSV; повторять, пока не напишет "saved"
  python simple.py predict --data /путь --test-dir /путь/test --template /путь/sample.csv --out outputs/simple.csv
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from clf_features import rec_feats
from common import ACTIONS, ACTIVE, CONTEXT, day_folds, load_all, score
from feats import DS, build

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_LOG = 18  # первые 18 признаков любого набора — лог-огибающие (3 масштаба x 6 каналов)

# Три семейства признаков. Каждое ошибается по-своему, поэтому их среднее лучше любого одного.
FAMILIES = {
    "envraw": dict(chdrop=0.15),  # огибающие + сырой сигнал, сеть сама учит фильтры
    "envtan": dict(chdrop=0.0),   # огибающие + ковариации каналов (tangent space)
    "envcorr": dict(chdrop=0.0),  # огибающие + корреляции каналов (не зависят от усиления)
}
TEACHER = dict(hidden=64, drop=0.2, steps=1000)
STUDENT = dict(hidden=96, drop=0.3, steps=1500)
CLF_FEATS = ["env", "envraw", "envcorr"]
CLF_STEPS = 800
SEEDS = 3
FAST_FOLD = 5  # --quick/--fast по умолчанию гоняют один фолд: день с 3 сессиями (24 записи)

# Ошибка ансамбля v2 по сессиям (results/per_session_v2.md) — ориентир для отчёта по части фолдов.
V2_SESSION = {"ses_b0096aca": 0.076, "ses_fa6b7158": 0.107, "ses_c2c7265f": 0.121, "ses_4d8a5f08": 0.082,
              "ses_ffa5aa31": 0.085, "ses_0ca10a67": 0.082, "ses_195ffcec": 0.075, "ses_2e837ba6": 0.094,
              "ses_77695b87": 0.075, "ses_75a236ea": 0.116}


# ---------------------------------------------------------------- утилиты

_FEATS = {}


def features(x, kind):
    """Признаки записи с кэшем: в CV одна и та же запись нужна во многих фолдах."""
    key = (id(x), kind)
    if key not in _FEATS:
        _FEATS[key] = build(x, kind)
    return _FEATS[key]


def onehot(action):
    v = np.zeros(8, np.float32)
    v[ACTIONS.index(action)] = 1
    return v


def finger_weights(action):
    """Веса углов в лоссе = веса метрики: активный палец 0.5, остальные по 0.125; для Fist/Open/Close по 0.2."""
    w = np.full(5, 0.2, np.float32)
    a = ACTIVE.get(action)
    if a is not None:
        w[:] = 0.125
        w[a] = 0.5
    return np.repeat(w, 4) / 4 * 20


def upsample(p, T):
    """Линейная интерполяция предсказаний с 50 Гц на 500 Гц."""
    t = np.arange(T) / DS
    i0 = np.clip(np.floor(t).astype(int), 0, len(p) - 1)
    i1 = np.clip(i0 + 1, 0, len(p) - 1)
    w = (t - np.floor(t))[:, None]
    return p[i0] * (1 - w) + p[i1] * w


def assign(P, per_action):
    """Венгерский алгоритм: каждое действие получает ровно per_action записей, максимум суммарной log-вероятности."""
    ri, ci = linear_sum_assignment(np.repeat(-np.log(P + 1e-9), per_action, 1))
    out = np.zeros(len(P), int)
    out[ri] = ci // per_action
    return out


def random_crops(X, Y, ids, L, rng):
    xb, yb = [], []
    for i in ids:
        s = rng.integers(0, len(X[i]) - L)
        xb.append(X[i][s:s + L])
        if Y is not None:
            yb.append(Y[i][s:s + L])
    return torch.stack(xb), (torch.stack(yb) if Y is not None else None)


def gain_shift(bs, nch, sigma, sd):
    """Случайный сдвиг лог-усиления каждого канала и всех вместе: имитация дрейфа электродов между днями."""
    g = torch.randn(bs, 1, 6, device=DEV) * sigma + torch.randn(bs, 1, 1, device=DEV) * sigma
    mask = np.zeros(nch, np.float32)
    mask[:N_LOG] = 1  # сдвиг только для лог-огибающих
    return g[:, :, torch.arange(nch, device=DEV) % 6] * torch.tensor(mask / sd, device=DEV)


# ---------------------------------------------------------------- сеть

class Block(nn.Module):
    """Residual-блок: dilated свёртка -> GroupNorm -> GELU -> 1x1 свёртка."""

    def __init__(self, c, dilation, drop, k=5):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=(k - 1) // 2 * dilation, dilation=dilation)
        self.norm = nn.GroupNorm(1, c)
        self.pw = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.pw(F.gelu(self.norm(self.conv(x)))))


class Regressor(nn.Module):
    """[признаки + one-hot действия] -> 5 TCN-блоков -> BiLSTM x2 -> [+ one-hot] -> 20 углов на каждый кадр."""

    def __init__(self, cin, hidden, drop):
        super().__init__()
        self.inp = nn.Conv1d(cin + 8, hidden, 1)
        self.tcn = nn.Sequential(*[Block(hidden, 2 ** i, drop) for i in range(5)])  # дилатации 1..16
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.out = nn.Linear(2 * hidden + 8, 20)

    def forward(self, x, a):  # x: (B, T, C), a: (B, 8)
        T = x.shape[1]
        h = torch.cat([x, a[:, None, :].expand(-1, T, -1)], 2).transpose(1, 2)
        h = self.tcn(self.inp(h)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.out(torch.cat([h, a[:, None, :].expand(-1, T, -1)], 2))


class Classifier(nn.Module):
    """6 TCN-блоков -> mean+std по времени -> 8 классов действия."""

    def __init__(self, cin, hidden=64, drop=0.3):
        super().__init__()
        self.inp = nn.Conv1d(cin, hidden, 1)
        self.tcn = nn.Sequential(*[Block(hidden, 2 ** i, drop) for i in range(6)])
        self.head = nn.Linear(2 * hidden, 8)

    def forward(self, x):  # x: (B, T, C)
        h = self.tcn(self.inp(x.transpose(1, 2)))
        return self.head(torch.cat([h.mean(2), h.std(2)], 1))


# ---------------------------------------------------------------- обучение

def train_regressor(X, Y, actions, weights, hidden, drop, steps, chdrop, seed, bs=32, crop=400):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allx = np.concatenate(X)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6  # нормировка по статистике обучающих данных
    Xt = [torch.tensor((x - mu) / sd, dtype=torch.float32, device=DEV) for x in X]
    Yt = [torch.tensor(y, dtype=torch.float32, device=DEV) for y in Y]
    At = torch.tensor(np.stack([onehot(a) for a in actions]), device=DEV)
    Wt = torch.tensor(np.stack(weights), device=DEV)
    nch = X[0].shape[1]
    net = Regressor(nch, hidden, drop).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    chan = torch.arange(nch, device=DEV) % 6
    net.train()
    for _ in range(steps):
        idx = rng.integers(0, len(X), bs)
        xb, yb = random_crops(Xt, Yt, idx, crop, rng)  # окна по 8 с
        if chdrop > 0:  # выкидываем случайные каналы целиком
            xb = xb * (torch.rand(bs, 1, 6, device=DEV) > chdrop).float()[:, :, chan]
        xb = xb + gain_shift(bs, nch, 0.5, sd) + 0.05 * torch.randn_like(xb)
        loss = ((net(xb, At[idx]) - yb).abs() * Wt[idx][:, None, :]).mean()  # L1 с весами метрики
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    net.eval()

    @torch.no_grad()
    def predict(x, action):  # вся запись целиком, 50 Гц
        xt = torch.tensor((x - mu) / sd, dtype=torch.float32, device=DEV)[None]
        return net(xt, torch.tensor(onehot(action), device=DEV)[None])[0].cpu().numpy()

    return predict


def train_classifier(X, labels, seed, bs=64, crop=500):
    steps = CLF_STEPS
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allx = np.concatenate(X)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    Xt = [torch.tensor((x - mu) / sd, dtype=torch.float32, device=DEV) for x in X]
    yt = torch.tensor(labels, device=DEV)
    net = Classifier(X[0].shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    net.train()
    for _ in range(steps):
        idx = rng.integers(0, len(X), bs)
        xb, _ = random_crops(Xt, None, idx, crop, rng)  # окна по 10 с
        xb = xb + gain_shift(bs, xb.shape[2], 0.3, sd) + 0.05 * torch.randn_like(xb)
        loss = F.cross_entropy(net(xb), yt[idx], label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()

    @torch.no_grad()
    def log_proba(x):  # усреднение по скользящим окнам всей записи
        xt = torch.tensor((x - mu) / sd, dtype=torch.float32, device=DEV)
        L = min(crop, len(xt))
        wins = torch.stack([xt[s:s + L] for s in range(0, len(xt) - L + 1, max(L // 4, 1))])
        return F.log_softmax(net(wins), 1).mean(0).cpu().numpy()

    return log_proba


# ---------------------------------------------------------------- шаги с сохранением на диск

class Pause(Exception):
    """Время вышло: готовые шаги сохранены, повторный запуск продолжит с этого места."""


DEADLINE = float("inf")


def step(path, desc, fn):
    """Один «батч» работы: если результат уже на диске — читаем его, иначе считаем и сохраняем."""
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    if time.time() > DEADLINE:
        raise Pause
    t0 = time.time()
    print(f"  {desc} ...", end=" ", flush=True)
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    os.replace(tmp, path)  # атомарно: прерванная запись не оставит битый файл
    print(f"{time.time() - t0:.0f} с", flush=True)
    return res


def classify(d, x_tr, act_tr, x_te, seeds):
    """Вероятности действия: геометрическое среднее CNN по 3 видам признаков и LogReg. 4 шага."""
    y = np.array([ACTIONS.index(a) for a in act_tr])

    def cnn(kind):
        Ftr = [features(x, kind) for x in x_tr]
        Fte = [features(x, kind) for x in x_te]
        per_seed = []
        for s in range(seeds):
            lp = train_classifier(Ftr, y, s)
            per_seed.append(np.stack([lp(f) for f in Fte]))
        return np.mean(per_seed, 0)

    def logreg():
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
        lr.fit(np.stack([rec_feats(x) for x in x_tr]), y)
        return np.log(lr.predict_proba(np.stack([rec_feats(x) for x in x_te])) + 1e-9)

    logps = [step(d / f"clf_{k}.pkl", f"классификатор CNN {k}", lambda k=k: cnn(k)) for k in CLF_FEATS]
    logps.append(step(d / "clf_logreg.pkl", "классификатор LogReg", logreg))
    P = np.exp(np.mean(logps, 0))
    return P / P.sum(1, keepdims=True)


def regress(d, x_tr, y_tr, act_tr, x_te, act_te, seeds, students=True):
    """Учитель (3 шага) -> метки на тесте -> студенты (3 шага) на трейне + тесте. Выход 50 Гц."""
    Y = [y[::DS] for y in y_tr]
    W = [finger_weights(a) for a in act_tr]

    def family_pred(kind, cfg, extra_y=None, seed0=0):
        Ftr = [features(x, kind) for x in x_tr]
        Fte = [features(x, kind) for x in x_te]
        X, Yf, A, Wf = Ftr, Y, list(act_tr), W
        if extra_y is not None:  # дистилляция: тестовые записи с метками учителя
            X, Yf = Ftr + Fte, Y + [p[:len(f)] for p, f in zip(extra_y, Fte)]
            A, Wf = A + list(act_te), W + [finger_weights(a) for a in act_te]
        models = [train_regressor(X, Yf, A, Wf, cfg["hidden"], cfg["drop"], cfg["steps"],
                                  FAMILIES[kind]["chdrop"], seed0 + s) for s in range(seeds)]
        return [np.mean([m(f, a) for m in models], 0) for f, a in zip(Fte, act_te)]

    teacher = [step(d / f"teacher_{k}.pkl", f"учитель {k}", lambda k=k: family_pred(k, TEACHER)) for k in FAMILIES]
    blend = [np.mean([t[i] for t in teacher], 0) for i in range(len(x_te))]
    if not students:
        return blend, blend
    student = [step(d / f"student_{k}.pkl", f"студент {k}", lambda k=k: family_pred(k, STUDENT, blend, 50))
               for k in FAMILIES]
    return blend, [np.mean([s[i] for s in student], 0) for i in range(len(x_te))]


# ---------------------------------------------------------------- режимы

def run_cv(a, work):
    meta, xs, ys = load_all()
    folds = list(enumerate(day_folds(meta)))
    todo = folds if a.fold is None else [folds[k] for k in a.fold]
    for k, (tr, va) in todo:
        d = work / "cv" / f"fold{k}"
        if (d / result_name(a)).exists():
            print(f"фолд {k}: уже готов", flush=True)
            continue
        print(f"фолд {k} (0..6): день {va.day_id.iloc[0]}, записей {len(va)}", flush=True)
        x_tr = [xs[r] for r in tr.recording_id]
        x_va = [xs[r] for r in va.recording_id]
        if a.true_actions:  # без классификатора: проверяем только регрессию (классификатор и так ~100%)
            acts = list(va.action)
        else:
            P = classify(d, x_tr, list(tr.action), x_va, a.seeds)
            acts = [ACTIONS[i] for i in assign(P, va.session_id.nunique())]
        t_pred, s_pred = regress(d, x_tr, [ys[r] for r in tr.recording_id], list(tr.action), x_va, acts,
                                 a.seeds, students=not a.no_students)
        step(d / result_name(a), "итог фолда", lambda: dict(recs=list(va.recording_id), acts=acts,
                                                           teacher=t_pred, final=s_pred))
    run_report(a, work)


def result_name(a):
    return "result_teacher_only.pkl" if a.no_students else "result.pkl"


def run_report(a, work):
    meta, _, ys = load_all()
    files = sorted((work / "cv").glob(f"fold*/{result_name(a)}"))
    if not files:
        print("Нет готовых фолдов. Сначала: python simple.py cv --fold 0 (с теми же --seeds/--quick/--no-students)")
        return
    true_act = dict(zip(meta.recording_id, meta.action))
    teach, final, correct = {}, {}, []
    for f in files:
        with open(f, "rb") as fh:
            res = pickle.load(fh)
        for r, act, tp, sp in zip(res["recs"], res["acts"], res["teacher"], res["final"]):
            teach[r] = upsample(tp, len(ys[r]))
            final[r] = upsample(sp, len(ys[r]))
            correct.append(act == true_act[r])
    s_t, _ = score(meta, teach, ys)
    s_f, df = score(meta, final, ys)
    sessions = df.session_id.unique()
    ref = np.mean([V2_SESSION.get(s, np.nan) for s in sessions])
    print(f"\nготово фолдов: {len(files)} из 7 ({', '.join(f.parent.name for f in files)}), сессий: {len(sessions)}")
    print(f"точность действия: {np.mean(correct):.4f}" + ("  (взяты истинные действия)" if a.true_actions else ""))
    print(f"wMAE учителя:      {s_t:.4f}")
    print(f"wMAE итог:         {s_f:.4f}" + ("" if len(files) == 7 else "   (не все фолды — оценка предварительная)"))
    if not np.isnan(ref):
        print(f"для сравнения: ансамбль v2 (23 модели x 3 сида, полное обучение) на этих же сессиях — {ref:.4f}")
    print(df.groupby("action").err.mean().round(3).to_string())


def run_predict(a, work):
    meta, xs, ys = load_all()
    template = pd.read_csv(a.template)
    angle_cols = [f"angle_{i:02d}" for i in range(20)]
    rids = list(dict.fromkeys(template.recording_id))
    d = work / "predict"
    d.mkdir(parents=True, exist_ok=True)
    ids_file = d / "recording_ids.txt"
    if ids_file.exists() and ids_file.read_text().split() != rids:
        raise SystemExit(f"В {d} сохранены шаги для другого теста. Удалите эту папку или задайте другой --work")
    ids_file.write_text("\n".join(rids))
    xt = [np.load(Path(a.test_dir) / f"{r}.npy") for r in rids]
    x_tr = [xs[r] for r in meta.recording_id]
    P = classify(d, x_tr, list(meta.action), xt, a.seeds)
    choice = assign(P, len(rids) // 8) if len(rids) % 8 == 0 else P.argmax(1)
    acts = [ACTIONS[i] for i in choice]
    _, preds = regress(d, x_tr, [ys[r] for r in meta.recording_id], list(meta.action), xt, acts, a.seeds,
                       students=not a.no_students)
    out = template.copy()
    for r, x, p in zip(rids, xt, preds):
        full = np.clip(upsample(p, len(x)), -np.pi, np.pi)
        rows = (template.recording_id == r).to_numpy()
        idx = template.loc[rows, "sample_index"].to_numpy()
        assert idx.min() >= CONTEXT and idx.max() < len(full)
        out.loc[rows, angle_cols] = full[idx]
    assert np.isfinite(out[angle_cols].to_numpy(np.float64)).all()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6f")
    info = pd.DataFrame({"recording_id": rids, "action": acts, "p_assigned": P[np.arange(len(rids)), choice],
                         "argmax": [ACTIONS[i] for i in P.argmax(1)]})
    info.to_csv(Path(a.out).with_suffix(".actions.csv"), index=False)
    print(info.to_string())
    print("saved", a.out, out.shape)


def find_data(given):
    """Папка с данными = та, где лежит train/metadata.csv. Если --data не задан, ищем в типичных местах."""
    if given:
        roots = [Path(given)]
    else:
        here = Path(__file__).resolve().parent
        roots = [Path.cwd() / "data", here / "data", Path.cwd(), here, Path("/data"), Path("/mnt/data"),
                 Path("/input"), Path("/kaggle/input"), Path.home() / "data"]
    for root in roots:
        if (root / "train" / "metadata.csv").exists():
            return root
    for root in roots:  # на уровень-два глубже: например /data/<имя_датасета>/train
        if root.is_dir():
            for meta in list(root.glob("*/train/metadata.csv")) + list(root.glob("*/*/train/metadata.csv")):
                return meta.parent.parent
    raise SystemExit("Не нашёл train/metadata.csv. Укажите папку с данными: --data /путь/к/папке "
                     "(внутри должны быть train/metadata.csv, train/signals/, train/targets/)")


def main(argv=None):
    global TEACHER, STUDENT, CLF_STEPS, DEADLINE
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["cv", "report", "predict"])
    p.add_argument("--data", help="папка с train/ (по умолчанию ищется автоматически)")
    p.add_argument("--test-dir", help="по умолчанию <data>/test")
    p.add_argument("--template", help="по умолчанию <data>/phase2_sample_submission.csv")
    p.add_argument("--out", default="outputs/simple.csv")
    p.add_argument("--work", default="work", help="куда сохранять готовые шаги")
    p.add_argument("--fold", type=int, nargs="+", choices=range(7), help="cv: только эти фолды (0..6)")
    p.add_argument("--time-limit", type=float, help="минуты: после них не начинать новый шаг")
    p.add_argument("--seeds", type=int, default=SEEDS)
    p.add_argument("--no-students", action="store_true", help="без дистилляции: только учитель")
    p.add_argument("--true-actions", action="store_true", help="cv: брать истинные действия, без классификатора")
    p.add_argument("--quick", action="store_true", help="проверка кода: 20 шагов, 1 сид, 1 фолд (минуты)")
    p.add_argument("--fast", action="store_true",
                   help="быстрая оценка на CPU: 1 сид, шагов в 4 раза меньше, истинные действия, 1 фолд (~30 мин)")
    a = p.parse_args(argv)
    if a.quick:
        TEACHER, STUDENT, CLF_STEPS = {**TEACHER, "steps": 20}, {**STUDENT, "steps": 20}, 20
        a.seeds = 1
    elif a.fast:
        TEACHER = {**TEACHER, "steps": TEACHER["steps"] // 4}
        STUDENT = {**STUDENT, "steps": STUDENT["steps"] // 4}
        CLF_STEPS //= 4
        a.seeds = 1
        a.true_actions = True
    if (a.quick or a.fast) and a.fold is None:
        a.fold = [FAST_FOLD]
    if DEV == "cpu":  # на CPU torch по умолчанию берёт все логические потоки, половина работает быстрее
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    DEADLINE = time.time() + a.time_limit * 60 if a.time_limit else float("inf")
    # Настройки входят в имя папки, чтобы шаги разных режимов (--quick, --fast, другое число сидов) не смешивались.
    mode = "_quick" if a.quick else "_fast" if a.fast else ""
    work = Path(a.work) / (f"seeds{a.seeds}{mode}" + ("_trueact" if a.true_actions and a.mode != "predict" else ""))
    common.DATA = find_data(a.data)  # load_all() читает данные отсюда
    print("device:", DEV, "| потоков CPU:", torch.get_num_threads(), "| data:", common.DATA, "| work:", work, flush=True)
    if a.mode == "predict":
        a.test_dir = a.test_dir or str(common.DATA / "test")
        a.template = a.template or str(common.DATA / "phase2_sample_submission.csv")
        for path in (a.test_dir, a.template):
            if not Path(path).exists():
                raise SystemExit(f"Нет {path}. Укажите --test-dir и --template явно")
        print("test:", a.test_dir, "| template:", a.template, flush=True)
    try:
        {"cv": run_cv, "report": run_report, "predict": run_predict}[a.mode](a, work)
    except Pause:
        print(f"\nЛимит времени. Готовые шаги сохранены в {work}. Запустите ту же команду ещё раз — продолжит с этого места.")


if __name__ == "__main__":
    main()
