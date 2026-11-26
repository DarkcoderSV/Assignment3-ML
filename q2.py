"""Q2: CNN vs MLP (reuses Q1a models), spatial disturbance (pixel / block permutation),
and a colour-vs-spatial-structure experiment (grayscale) with error/confidence analysis."""
import numpy as np
import torch
import torch.nn.functional as F

from common import (CLASSES, apply_transform, device, load_config, load_data, load_json,
                    make_transform, n_params, path, predict, save_json, set_seed, strip,
                    train_classifier)
from q1 import build


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def analyse(probs, y):
    pred, conf = probs.argmax(1), probs.max(1)
    correct = pred == y
    return {"test_acc": float(correct.mean()),
            "per_class_acc": [float(correct[y == c].mean()) for c in range(10)],
            "mean_conf": float(conf.mean()),
            "mean_conf_correct": float(conf[correct].mean()),
            "mean_conf_wrong": float(conf[~correct].mean()),
            "frac_wrong_conf_gt_0.9": float((conf[~correct] > 0.9).mean()),
            "ece": ece(conf, correct),
            "nll": float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()),
            "confusion": np.bincount(y * 10 + pred, minlength=100).reshape(10, 10).tolist()}


def shift_probe(model, x, y, shifts=(0, 1, 2, 4)):
    """Test-time diagnostic only: accuracy when test images are translated by k pixels
    (zero fill), averaged over the 4 axis directions."""
    out = {}
    for k in shifts:
        accs = []
        for dy, dx in ([(0, 0)] if k == 0 else [(k, 0), (-k, 0), (0, k), (0, -k)]):
            xs = torch.roll(x, (dy, dx), (2, 3))
            if dy > 0: xs[:, :, :dy] = 0
            if dy < 0: xs[:, :, dy:] = 0
            if dx > 0: xs[:, :, :, :dx] = 0
            if dx < 0: xs[:, :, :, dx:] = 0
            accs.append((predict(model, xs).argmax(1) == y).float().mean().item())
        out[str(k)] = float(np.mean(accs))
    return out


def main():
    cfg = load_config()
    base = load_data(cfg)
    q1 = load_json("results", "q1a", "summary.json")
    summary = {"models": {k: {"params": q1[k]["params"], "lr": q1[k]["lr"]}
                          for k in ["MLP-M", "CNN"]}}
    y_test = base["test"][1].cpu().numpy()
    for cond in ["clean", "pixel_perm", "block_perm", "gray"]:
        data = apply_transform(base, make_transform(cond, cfg["seed"]))
        for name in ["MLP-M", "CNN"]:
            m = build(name, data)
            if cond == "clean":  # reused from Q1(a), not retrained
                m.load_state_dict(torch.load(path("results", "ckpt", f"q1a_{name}.pt")))
                h = load_json("results", "q1a", f"{name}_lr{q1[name]['lr']}.json")
            else:
                set_seed(cfg["seed"])
                h = train_classifier(m, data, q1[name]["lr"], cfg["clf_epochs"],
                                     cfg["clf_weight_decay"], cfg["batch_size"],
                                     seed=cfg["seed"], log=f"{cond} {name}")
                h = strip(h)
                save_json(h, "results", "q2", f"{cond}_{name}.json")
                torch.save(m.state_dict(), path("results", "ckpt", f"q2_{cond}_{name}.pt"))
            probs = F.softmax(predict(m, data["test"][0]), 1).cpu().numpy()
            np.save(path("results", "q2", f"probs_{cond}_{name}.npy"), probs.astype(np.float32))
            r = analyse(probs, y_test)
            r.update(val_acc=h["best_val_acc"], best_epoch=h["best_epoch"],
                     train_acc_at_best=h["train_acc"][h["best_epoch"] - 1], params=n_params(m))
            if cond == "clean":
                r["shift_probe"] = shift_probe(m, *data["test"])
            summary[f"{cond}/{name}"] = r
            print(cond, name, {k: r[k] for k in ["val_acc", "test_acc", "mean_conf", "ece"]},
                  flush=True)
    summary["classes"] = CLASSES
    save_json(summary, "results", "q2", "summary.json")


if __name__ == "__main__":
    main()
