"""Q1: (a) why MLPs do poorly — capacity/optimisation vs missing spatial prior;
(b) label-noise memorisation trajectory."""
import argparse

import numpy as np
import torch

from common import (CNN, MLP, accuracy, channel_stats, device, load_config, load_data,
                    n_params, path, save_json, set_seed, strip, train_classifier)

# Comparison models. MLP-M and CNN are parameter matched (~1.7M); MLP-S/L are 1/4x and 4x.
MODELS = {
    "MLP-S": lambda m, s: MLP([128, 128], m, s),
    "MLP-M": lambda m, s: MLP([480, 480], m, s),
    "MLP-L": lambda m, s: MLP([1536, 1536], m, s),
    "CNN": lambda m, s: CNN((64, 160, 320), m, s),
}


def build(name, data):
    mean, std = channel_stats(data["train"][0])
    return MODELS[name](mean, std).to(device())


def run_a(cfg, data):
    results = {}
    for name in MODELS:
        runs = {}
        for lr in cfg["clf_lr_grid"]:
            set_seed(cfg["seed"])
            m = build(name, data)
            h = train_classifier(m, data, lr, cfg["clf_epochs"], cfg["clf_weight_decay"],
                                 cfg["batch_size"], seed=cfg["seed"], log=f"{name} lr={lr}")
            runs[lr] = (h, m)
            save_json(strip(h), "results", "q1a", f"{name}_lr{lr}.json")
        lr_best = max(runs, key=lambda k: runs[k][0]["best_val_acc"])
        h, m = runs[lr_best]
        torch.save(m.state_dict(), path("results", "ckpt", f"q1a_{name}.pt"))
        results[name] = {"params": n_params(m), "lr": lr_best, "best_epoch": h["best_epoch"],
                         "val_acc": h["best_val_acc"],
                         "train_acc_at_best": h["train_acc"][h["best_epoch"] - 1],
                         "final_train_acc": h["train_acc"][-1],
                         "test_acc": accuracy(m, *data["test"]),
                         "lr_sweep": {str(k): v[0]["best_val_acc"] for k, v in runs.items()}}
        print(name, results[name], flush=True)

    # Augmentation control: injects translation invariance through data, not architecture.
    lr = results["MLP-M"]["lr"]
    set_seed(cfg["seed"])
    m = build("MLP-M", data)
    h = train_classifier(m, data, lr, cfg["clf_epochs"], cfg["clf_weight_decay"],
                         cfg["batch_size"], aug=True, seed=cfg["seed"], log="MLP-M+aug")
    save_json(strip(h), "results", "q1a", "MLP-M+aug.json")
    results["MLP-M+aug"] = {"params": n_params(m), "lr": lr, "best_epoch": h["best_epoch"],
                            "val_acc": h["best_val_acc"],
                            "train_acc_at_best": h["train_acc"][h["best_epoch"] - 1],
                            "final_train_acc": h["train_acc"][-1],
                            "test_acc": accuracy(m, *data["test"])}

    # Seed repeats of the two matched models at their selected LR (variance estimate).
    for name in ["MLP-M", "CNN"]:
        vals, tests = [results[name]["val_acc"]], [results[name]["test_acc"]]
        for s in (1, 2):
            set_seed(cfg["seed"] + s)
            m = build(name, data)
            h = train_classifier(m, data, results[name]["lr"], cfg["clf_epochs"],
                                 cfg["clf_weight_decay"], cfg["batch_size"],
                                 seed=cfg["seed"] + s, log=f"{name} seed+{s}")
            vals.append(h["best_val_acc"])
            tests.append(accuracy(m, *data["test"]))
        results[name]["seed_val"] = vals
        results[name]["seed_test"] = tests
    save_json(results, "results", "q1a", "summary.json")


def run_b(cfg, data):
    xtr, ytr = data["train"]
    n = len(ytr)
    out = {}
    for p in cfg["mem_noise_rates"]:
        rng = np.random.RandomState(cfg["seed"] % (2 ** 32) + int(p * 100))
        noisy = np.sort(rng.choice(n, int(round(p * n)), replace=False))
        y_true = ytr.cpu().numpy()
        y_new = y_true.copy()
        if p == 1.0:
            y_new = rng.randint(0, 10, n)  # fully random labels (no structure at all)
        else:
            # mislabelled examples get a uniformly random *wrong* class
            y_new[noisy] = (y_true[noisy] + rng.randint(1, 10, len(noisy))) % 10
        y_new_t = torch.from_numpy(y_new).long().to(ytr.device)
        clean = np.setdiff1d(np.arange(n), noisy)
        sub = lambda a: torch.from_numpy(a[:5000]).to(ytr.device)
        track = {}
        if len(clean):
            ci = sub(clean)
            track["clean_subset_acc"] = (xtr[ci], ytr[ci])
        if len(noisy):
            ni = sub(noisy)
            track["noisy_fit_given"] = (xtr[ni], y_new_t[ni])   # memorisation
            track["noisy_fit_true"] = (xtr[ni], ytr[ni])        # generalising to own true label
        set_seed(cfg["seed"])
        m = MLP([1536, 1536], *channel_stats(xtr)).to(device())
        h = train_classifier(m, data, cfg["mem_lr"], cfg["mem_epochs"], 0.0, cfg["batch_size"],
                             seed=cfg["seed"], train_labels=y_new_t, schedule="constant",
                             track=track, log=f"mem p={p}")
        h["test_acc_best_val_epoch"] = accuracy(m, *data["test"])
        m.load_state_dict(h["final_state"])
        h["test_acc_final_epoch"] = accuracy(m, *data["test"])
        h["params"] = n_params(m)
        h["noise_rate"] = p
        out[str(p)] = strip(h)
        save_json(out[str(p)], "results", "q1b", f"mem_p{p}.json")
        print(p, h["best_epoch"], h["best_val_acc"], h["test_acc_best_val_epoch"],
              h["test_acc_final_epoch"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    data = load_data(cfg)
    (run_a if args.part == "a" else run_b)(cfg, data)
