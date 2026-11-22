import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.family"] = ["Avenir Next", "Arial Unicode MS"]
MONO = ["Source Code Pro Medium", "Arial Unicode MS"]

P = dict(
    emb="#FDE9D9", conv="#DCE9F7", norm="#FFF4C2", act="#DFF0D8", drop="#EDEDED", add="#F5F5F5",
    lstm="#E8DAEF", lin="#FAD7C8", out="#F8CBCB", cls="#FCE3C4", hung="#F6B98A", teach="#D3E4F5",
    stud="#E1D8F0", pseudo="#ECECEC", frame="#F7F7F9", edge="#2E3440", accent="#B03A2E", blue="#1F4E8C",
)


def rbox(ax, x, y, w, h, text, fc, fs=11, bold=False, ec=None, lw=1.3, rs=0.9, tc="#111", sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                                fc=fc, ec=ec or P["edge"], lw=lw, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "medium", color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 2.2, color="#444", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "medium", color=tc, zorder=4, linespacing=1.3)
    return (x, y, w, h)


def arr(ax, p, q, color=None, lw=1.5, rad=0.0, style="-|>", ms=13, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color or P["edge"], lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def poly(ax, pts, color=None, lw=1.5, head=True):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color or P["edge"], lw=lw, zorder=2, solid_capstyle="round")
    if head:
        arr(ax, pts[-2], pts[-1], color=color, lw=lw)
    else:
        ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color=color or P["edge"], lw=lw, zorder=2)


def plus(ax, x, y, r=1.25):
    ax.add_patch(Circle((x, y), r, fc="white", ec=P["edge"], lw=1.4, zorder=4))
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color=P["edge"], lw=1.4, zorder=5)
    ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color=P["edge"], lw=1.4, zorder=5)


def concat(ax, x, y, r=1.25):
    ax.add_patch(Circle((x, y), r, fc="white", ec=P["edge"], lw=1.4, zorder=4))
    ax.text(x, y - 0.05, "⊕", ha="center", va="center", fontsize=13, zorder=5)


def frame(ax, x, y, w, h, title, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=P["frame"], ec="#B8BCC6", lw=1.2, zorder=0))
    ax.text(x + w / 2, y + h + 2.2, title, ha="center", va="bottom", fontsize=15, fontweight="bold", color="#1B1F24")
    if sub:
        ax.text(x + w / 2, y + h + 0.6, sub, ha="center", va="bottom", fontsize=10.5, color="#555")


fig = plt.figure(figsize=(19, 13.2), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 190)
ax.set_ylim(0, 132)
ax.axis("off")

ax.text(95, 129.5, "PULSE — model architecture", ha="center", va="top", fontsize=22, fontweight="bold", color="#111")
ax.text(95, 125.2, "6-channel surface EMG (500 Hz)  →  20 finger-joint angles per frame", ha="center", va="top", fontsize=12.5, color="#444")

# ---------------- column 1: action classifier ----------------
frame(ax, 4, 14, 46, 96, "① Action classifier", "recording-level · 8 classes")
cx, cw = 9, 36
rbox(ax, cx, 18, cw, 7, "EMG features", P["emb"], sub="env · envraw · envcorr …  (T, C)")
rbox(ax, cx, 30, cw, 6, "Conv1d 1×1  →  64", P["conv"])
blk_y = 40
ax.add_patch(FancyBboxPatch((cx - 1.5, blk_y - 1.5), cw + 3, 22, boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec="#8F96A3", lw=1.1, ls=(0, (4, 3)), zorder=1))
rbox(ax, cx, blk_y + 13, cw, 6, "Dilated Conv  k=5,  d = 1…32", P["conv"], fs=10.5)
rbox(ax, cx, blk_y + 6.5, cw, 5, "GroupNorm · GELU", P["norm"], fs=10.5)
rbox(ax, cx, blk_y, cw, 5, "Conv 1×1 · Dropout · Add", P["add"], fs=10.5)
ax.text(cx + cw + 2.2, blk_y + 9.5, "×6", fontsize=14, fontweight="bold", color=P["blue"], va="center")
rbox(ax, cx, 67, cw, 6, "Mean ⊕ Std pooling over time", P["lstm"], fs=10.5)
rbox(ax, cx, 77, cw, 6, "Linear → Softmax (8)", P["lin"], fs=10.5)
rbox(ax, cx, 87, cw, 7, "Ensemble · geometric mean", P["cls"], fs=10.5, sub="7 CNN × 3 seeds  +  LogReg")
rbox(ax, cx, 98, cw, 8, "Hungarian assignment", P["hung"], bold=True, fs=11.5, sub="each action exactly 4× in test")
for a, b in [(25, 30), (36, blk_y - 1.5), (blk_y + 20.5, 67), (73, 77), (83, 87), (94, 98)]:
    arr(ax, (cx + cw / 2, a), (cx + cw / 2, b))
ax.text(27, 15.6, "accuracy 98.75 %  →  100 %", ha="center", fontsize=10, color=P["accent"], fontweight="bold")

# ---------------- column 2: regressor ----------------
frame(ax, 58, 3.5, 72, 106.5, "② Pose regressor", "action-conditioned TCN–BiLSTM · sequence-to-sequence")
rx, rw = 66, 40
rbox(ax, rx, 9, rw, 7.5, "EMG features  x", P["emb"], sub="(B, C, T) · 50 Hz")
rbox(ax, rx + rw + 4, 9, 16, 7.5, "One-hot  a", P["emb"], sub="(B, 8), × T")
concat(ax, rx + rw / 2, 20.5)
arr(ax, (rx + rw / 2, 16.5), (rx + rw / 2, 19.25))
poly(ax, [(rx + rw + 9, 16.5), (rx + rw + 9, 20.5), (rx + rw / 2 + 1.25, 20.5)])
rbox(ax, rx, 24, rw, 5.5, "Conv1d 1×1  →  H", P["conv"], sub=None)
ax.text(rx + rw + 2, 26.75, "H = 64 / 96", fontsize=9.5, color="#555", va="center", family=MONO)
arr(ax, (rx + rw / 2, 21.75), (rx + rw / 2, 24))

tb_y0, tb_h = 33, 36
ax.add_patch(FancyBboxPatch((rx - 3, tb_y0), rw + 6, tb_h, boxstyle="round,pad=0,rounding_size=1.8", fc="white", ec="#8F96A3", lw=1.2, ls=(0, (4, 3)), zorder=1))
ax.text(rx + rw + 5.0, tb_y0 + tb_h / 2 + 3, "× 5", fontsize=18, fontweight="bold", color=P["blue"], va="center")
ax.text(rx + rw + 5.0, tb_y0 + tb_h / 2 - 2.5, "dilation\nd = 1, 2, 4, 8, 16", fontsize=9, color=P["blue"], va="center", linespacing=1.3)
ax.text(rx + rw + 5.0, tb_y0 + tb_h / 2 - 9.5, "receptive field\n≈ ±1.2 s", fontsize=9, color=P["blue"], va="center", linespacing=1.3)
sub = [("Dilated Conv1d  k = 5, d", P["conv"]), ("GroupNorm", P["norm"]), ("GELU", P["act"]), ("Conv1d 1×1", P["conv"]), ("Dropout  p = 0.2", P["drop"])]
sy = tb_y0 + 3.2
ys = []
for t, c in sub:
    rbox(ax, rx + 4, sy, rw - 8, 4.2, t, c, fs=10.3)
    ys.append(sy)
    sy += 5.6
for a, b in zip(ys[:-1], ys[1:]):
    arr(ax, (rx + rw / 2, a + 4.2), (rx + rw / 2, b))
arr(ax, (rx + rw / 2, 29.5), (rx + rw / 2, ys[0]))
plus(ax, rx + rw / 2, tb_y0 + tb_h - 3.2)
arr(ax, (rx + rw / 2, ys[-1] + 4.2), (rx + rw / 2, tb_y0 + tb_h - 4.45))
poly(ax, [(rx + rw / 2, 31.2), (rx + 0.5, 31.2), (rx + 0.5, tb_y0 + tb_h - 3.2), (rx + rw / 2 - 1.25, tb_y0 + tb_h - 3.2)], color=P["accent"])
ax.text(rx + 1.2, tb_y0 + tb_h / 2 + 6, "residual", fontsize=9.5, color=P["accent"], rotation=90, va="center")

ly = tb_y0 + tb_h + 4
arr(ax, (rx + rw / 2, tb_y0 + tb_h - 1.95), (rx + rw / 2, ly))
rbox(ax, rx, ly, rw, 10.5, "", P["lstm"])
ax.text(rx + rw / 2, ly + 7.9, "BiLSTM  × 2 layers", ha="center", va="center", fontsize=11.5, fontweight="bold", zorder=4)
for k, lab in enumerate(["→  forward LSTM", "←  backward LSTM"]):
    rbox(ax, rx + 3 + k * 17.5, ly + 1.4, 16.5, 4.4, lab, "white", fs=9.2, lw=1)
ax.text(rx + rw + 2, ly + 5.5, "(B, T, 2H)", fontsize=9.5, color="#555", va="center", family=MONO)

cy = ly + 14
arr(ax, (rx + rw / 2, ly + 10.5), (rx + rw / 2, cy - 1.25))
concat(ax, rx + rw / 2, cy)
poly(ax, [(rx + rw + 15, 16.5), (rx + rw + 15, cy), (rx + rw / 2 + 1.25, cy)])
rbox(ax, rx, cy + 3.5, rw, 5.5, "Linear  →  20", P["lin"], fs=11)
arr(ax, (rx + rw / 2, cy + 1.25), (rx + rw / 2, cy + 3.5))
rbox(ax, rx, cy + 12, rw, 8, "Joint angles  θ̂", P["out"], bold=True, fs=12, sub="(B, T, 20) → 500 Hz, clip [−π, π]")
arr(ax, (rx + rw / 2, cy + 9), (rx + rw / 2, cy + 12))

poly(ax, [(cx + cw, 102), (54, 102), (54, 5.6), (rx + rw + 12, 5.6), (rx + rw + 12, 9)], color=P["hung"], lw=2.2)
ax.text(54.8, 60, "assigned action", fontsize=10, color="#A0522D", rotation=90, va="center", fontweight="bold")

# ---------------- column 3: training & distillation ----------------
frame(ax, 138, 14, 48, 96, "③ Training & distillation", "transductive · uses test EMG only")
tx, tw = 142, 36
rbox(ax, tx, 98, tw, 8, "Loss = competition metric", P["out"], fs=11, bold=True, sub="weighted L1 · active finger 0.5")
rbox(ax, tx, 87, tw, 7.5, "Gain augmentation  σ = 0.5", P["act"], fs=10.5, sub="per-channel log-gain shift · ch-dropout")
rbox(ax, tx, 74, tw, 9, "Teacher ensemble", P["teach"], bold=True, fs=11.5, sub="7 families · 23 configs × 3 seeds · 0.0911")
rbox(ax, tx, 61, tw, 9, "Pseudo-labels on test EMG", P["pseudo"], fs=11, sub="train (true θ)  +  test (teacher θ̂)")
rbox(ax, tx, 48, tw, 9, "Students · round 1", P["stud"], bold=True, fs=11.5, sub="11 configs × 3 seeds · 0.0894")
rbox(ax, tx, 35, tw, 9, "Students · round 2", P["stud"], bold=True, fs=11.5, sub="7 configs × 3 seeds · 0.0889")
rbox(ax, tx, 20, tw, 10, "Final prediction", P["out"], bold=True, fs=12.5, sub="hidden test  0.0821 rad · 1st place", tc=P["accent"])
for a, b in [(74, 70), (61, 57), (48, 44), (35, 30)]:
    arr(ax, (tx + tw / 2, a), (tx + tw / 2, b))
poly(ax, [(tx + tw, 52.5), (tx + tw + 2.5, 52.5), (tx + tw + 2.5, 65.5), (tx + tw, 65.5)], color=P["blue"])
ax.text(tx + tw + 3.2, 59, "new\nteacher", fontsize=8.8, color=P["blue"], va="center", linespacing=1.2)
ax.text(tx + tw / 2, 84.6, "applied to every regressor", fontsize=9, color="#666", ha="center", style="italic")

ax.text(95, 1.2, "Leave-one-day-out CV wMAE 0.0889 rad  ·  hidden test 0.0821 rad (1st place)  ·  organiser Ridge baseline 0.346",
        ha="center", fontsize=11, color="#333")

for ext in ("png", "pdf", "svg"):
    fig.savefig(f"/Users/ismailgaleev/Downloads/PULSE_model_architecture.{ext}", dpi=220)
print("ok")
