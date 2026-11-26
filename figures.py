"""Generates every figure in the report from results/ (no training)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from common import CLASSES, ROOT, load_json, path

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"MLP-S": "#9ecae1", "MLP-M": "#3182bd", "MLP-L": "#08519c", "CNN": "#e6550d",
     "MLP-M+aug": "#31a354"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(path("figures", name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def q1a():
    s = load_json("results", "q1a", "summary.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    for name in ["MLP-S", "MLP-M", "MLP-L", "CNN", "MLP-M+aug"]:
        fn = "MLP-M+aug.json" if name == "MLP-M+aug" else f"{name}_lr{s[name]['lr']}.json"
        h = load_json("results", "q1a", fn)
        lab = f"{name} ({s[name]['params'] / 1e6:.2f}M)"
        ax[0].plot(h["epoch"], h["train_acc"], color=C[name], label=lab)
        ax[1].plot(h["epoch"], h["val_acc"], color=C[name], label=lab)
    ax[0].set_title("training accuracy"); ax[1].set_title("validation accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[0].set_ylabel("accuracy"); ax[1].legend(loc="lower right")
    save(fig, "q1a_curves.png")


def q1b():
    rates = load_json("results", "q1b", "mem_p0.4.json")
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.3))
    h = rates
    e = h["epoch"]
    ax[0].plot(e, h["val_acc"], "k", label="val acc (clean labels)")
    ax[0].plot(e, h["clean_subset_acc"], color="#3182bd", label="train: correctly-labelled")
    ax[0].plot(e, h["noisy_fit_given"], color="#e6550d", label="train: mislabelled, fit to noisy label")
    ax[0].plot(e, h["noisy_fit_true"], color="#31a354", ls="--", label="train: mislabelled, predicts true label")
    ax[0].axvline(h["best_epoch"], color="gray", ls=":", lw=1)
    ax[0].set_title("40% label noise, MLP-L"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy")
    ax[0].legend(loc="upper left", fontsize=5); ax[0].set_xscale("log")
    for p, col in zip(["0.0", "0.2", "0.4", "0.8", "1.0"],
                      ["#000000", "#3182bd", "#e6550d", "#756bb1", "#969696"]):
        h = load_json("results", "q1b", f"mem_p{p}.json")
        ax[1].plot(h["epoch"], h["val_acc"], color=col, label=f"p={p}")
        if "noisy_fit_given" in h:
            ax[2].plot(h["epoch"], h["noisy_fit_given"], color=col, label=f"p={p}")
    ax[1].set_title("validation accuracy"); ax[2].set_title("memorised fraction of noisy labels")
    for a in ax[1:]:
        a.set_xlabel("epoch"); a.set_xscale("log"); a.legend()
    for a in ax:
        a.grid(alpha=.3)
    save(fig, "q1b_memorisation.png")


def q2():
    s = load_json("results", "q2", "summary.json")
    y = np.array(torchvision.datasets.CIFAR10(os.path.join(ROOT, "data"), train=False).targets)
    # reliability diagrams
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.2))
    bins = np.linspace(0, 1, 11)
    for k, cond in enumerate(["clean", "pixel_perm", "gray"]):
        for name in ["MLP-M", "CNN"]:
            p = np.load(path("results", "q2", f"probs_{cond}_{name}.npy"))
            conf, corr = p.max(1), p.argmax(1) == y
            mids, accs = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() > 20:
                    mids.append(conf[m].mean()); accs.append(corr[m].mean())
            ax[k].plot(mids, accs, "o-", ms=2.5, color=C[name],
                       label=f"{name} acc={corr.mean():.3f} ECE={s[f'{cond}/{name}']['ece']:.3f}")
        ax[k].plot([0, 1], [0, 1], "k:", lw=.8)
        ax[k].set_title(cond); ax[k].set_xlabel("confidence"); ax[k].legend(loc="upper left", fontsize=5.5)
        ax[k].grid(alpha=.3)
    ax[0].set_ylabel("accuracy")
    save(fig, "q2_reliability.png")
    # per-class accuracy change under grayscale
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    w = 0.38
    for k, name in enumerate(["MLP-M", "CNN"]):
        d = np.array(s[f"gray/{name}"]["per_class_acc"]) - np.array(s[f"clean/{name}"]["per_class_acc"])
        ax.bar(np.arange(10) + (k - .5) * w, d, w, color=C[name], label=name)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_ylabel(r"$\Delta$ test acc (gray $-$ clean)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    save(fig, "q2_gray_perclass.png")


def q3():
    s = load_json("results", "q3", "summary.json")
    ex = np.load(path("results", "q3", "recon_examples.npz"))
    keys = ["clean", "AE(clean)", "noisy", "AE(noisy)", "DAE(noisy)"]
    fig, ax = plt.subplots(len(keys), 8, figsize=(4.6, 3.1))
    for r, k in enumerate(keys):
        for c in range(8):
            ax[r, c].imshow(ex[k][c].transpose(1, 2, 0).clip(0, 1)); ax[r, c].axis("off")
        ax[r, 0].text(-4, 16, k, ha="right", va="center", fontsize=6.5)
    save(fig, "q3_recons.png")
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for d, col in zip(s["a"].keys(), ["#9ecae1", "#3182bd", "#08519c"]):
        if d in ("selected", "selected_test_mse"):
            continue
        h = load_json("results", "q3", f"ae{d}_hist.json")
        ax.plot(h["epoch"], h["val_mse"], color=col, label=f"d={d} val")
        ax.plot(h["epoch"], h["train_mse"], color=col, ls="--", lw=.8)
    h = load_json("results", "q3", "dae_hist.json")
    ax.plot(h["epoch"], h["val_mse"], color="#e6550d", label="DAE val (noisy in)")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend(); ax.grid(alpha=.3)
    save(fig, "q3_ae_curves.png")


def q4():
    s = load_json("results", "q4", "summary.json")
    kernels = sorted({r["kernel"] for r in s["sweep"]}, key=["linear", "rbf", "poly"].index)
    fig, ax = plt.subplots(1, len(kernels), figsize=(6.8, 2.1), sharey=True)
    for a, k in zip(ax, kernels):
        rs = sorted([r for r in s["sweep"] if r["kernel"] == k], key=lambda r: r["C"])
        Cs = [r["C"] for r in rs]
        a.plot(Cs, [r["train_acc"] for r in rs], "s--", ms=3, color="#9ecae1", label="train acc")
        a.plot(Cs, [r["val_acc"] for r in rs], "o-", ms=3, color="#08519c", label="val acc")
        a.plot(Cs, [r["sv_frac"] for r in rs], "^-", ms=3, color="#e6550d", label="SV fraction")
        a.set_xscale("log"); a.set_title(f"{k} kernel"); a.set_xlabel("C"); a.grid(alpha=.3)
    ax[0].legend(); ax[0].set_ylim(0, 1.02)
    save(fig, "q4_sweep.png")

    tr = torchvision.datasets.CIFAR10(os.path.join(ROOT, "data"), train=True).data
    te = torchvision.datasets.CIFAR10(os.path.join(ROOT, "data"), train=False).data
    hp = s["hard_pairs"]
    fig, ax = plt.subplots(len(hp) * 2, 6, figsize=(4.8, 1.0 * 2 * len(hp)))
    for r, p in enumerate(hp):
        svs, seen = [], set()
        for it in p["violating_svs"][:3] + p["margin_svs"]:
            if it["idx"] not in seen:
                seen.add(it["idx"]); svs.append(it)
        rows = [("SV", svs[:6], tr),
                ("test", p["near_test"][:6], te)]
        for q, (kind, items, src) in enumerate(rows):
            a_row = ax[2 * r + q]
            for c in range(6):
                a_row[c].axis("off")
                if c < len(items):
                    it = items[c]
                    a_row[c].imshow(src[it["idx"]])
                    t = f"{it['label'][:4]}" + (f"→{it['pred'][:4]}" if kind == "test" else "")
                    a_row[c].set_title(f"{t} {it['score']:+.3f}" if kind == "test" else f"{t} {it['score']:+.2f}",
                                      fontsize=5, pad=1)
            a_row[0].text(-6, 16, f"{p['pair'][0][:4]}/{p['pair'][1][:4]}\n{kind}", ha="right",
                          va="center", fontsize=6)
    save(fig, "q4_pairs.png")


if __name__ == "__main__":
    for f in (q1a, q1b, q2, q3, q4):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
