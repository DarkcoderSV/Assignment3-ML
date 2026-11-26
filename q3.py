"""Q3: (a) three conv autoencoders differing only in bottleneck size; (b) denoising AE
evaluated through the fixed Q1(a) CNN; (c) frozen-representation probes vs reconstruction."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (ConvAE, add_noise, device, load_config, load_data, load_json, n_params,
                    path, predict, save_json, set_seed)
from q1 import build


@torch.no_grad()
def recon(ae, x, bs=1000):
    ae.eval()
    return torch.cat([ae(x[i:i + bs]) for i in range(0, len(x), bs)])


@torch.no_grad()
def encode(ae, x, bs=1000):
    ae.eval()
    return torch.cat([ae.encode(x[i:i + bs]) for i in range(0, len(x), bs)])


def mse(a, b):
    return F.mse_loss(a, b).item()


def train_ae(ae, data, cfg, sigma=0.0, log=""):
    """sigma=0: plain AE. sigma>0: denoising AE, fresh Gaussian noise per batch on the inputs,
    clean images as targets. Labels are never used. Keeps best-validation-MSE state."""
    xtr, xva = data["train"][0], data["val"][0]
    g = torch.Generator().manual_seed(cfg["seed"] + 99)
    xva_in = add_noise(xva, sigma, g) if sigma > 0 else xva
    opt = torch.optim.Adam(ae.parameters(), lr=cfg["ae_lr"])
    bs, E = cfg["batch_size"], cfg["ae_epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["ae_lr"], total_steps=E * ((len(xtr) + bs - 1) // bs),
                                                pct_start=0.15)
    hist = {"epoch": [], "train_mse": [], "val_mse": []}
    best, t0 = 1e9, time.time()
    for ep in range(1, E + 1):
        ae.train()
        perm = torch.randperm(len(xtr)).to(xtr.device)
        tot = 0.0
        for i in range(0, len(xtr), bs):
            xb = xtr[perm[i:i + bs]]
            inp = add_noise(xb, sigma) if sigma > 0 else xb
            loss = F.mse_loss(ae(inp), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
        v = mse(recon(ae, xva_in), xva)
        hist["epoch"].append(ep); hist["train_mse"].append(tot / len(xtr)); hist["val_mse"].append(v)
        if v < best:
            best, best_ep = v, ep
            state = {k: t.clone() for k, t in ae.state_dict().items()}
        print(f"[{log}] ep {ep} train {tot / len(xtr):.5f} val {v:.5f} ({time.time() - t0:.0f}s)",
              flush=True)
    ae.load_state_dict(state)
    hist["best_val_mse"], hist["best_epoch"] = best, best_ep
    return hist


def linear_probe(ztr, ytr, zva, yva, zte, yte, wds=(0.0, 1e-4, 1e-3, 1e-2), epochs=60):
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-6
    ztr, zva, zte = (ztr - mu) / sd, (zva - mu) / sd, (zte - mu) / sd
    best = None
    for wd in wds:
        torch.manual_seed(0)
        lin = nn.Linear(ztr.shape[1], 10).to(ztr.device)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
        for _ in range(epochs):
            perm = torch.randperm(len(ztr)).to(ztr.device)
            for i in range(0, len(ztr), 512):
                idx = perm[i:i + 512]
                loss = F.cross_entropy(lin(ztr[idx]), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            va = (lin(zva).argmax(1) == yva).float().mean().item()
            te = (lin(zte).argmax(1) == yte).float().mean().item()
            tr = (lin(ztr).argmax(1) == ytr).float().mean().item()
        if best is None or va > best["val_acc"]:
            best = {"wd": wd, "train_acc": tr, "val_acc": va, "test_acc": te}
    return best


def knn_probe(ztr, ytr, zva, yva, zte, yte, ks=(5, 20, 50)):
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-6
    f = lambda z: F.normalize((z - mu) / sd, dim=1)
    ztr_n = f(ztr)

    def preds(z, k):
        z = f(z)
        out = []
        for i in range(0, len(z), 1000):
            sim = z[i:i + 1000] @ ztr_n.T
            nb = sim.topk(k, dim=1).indices
            votes = F.one_hot(ytr[nb], 10).sum(1)
            out.append(votes.argmax(1))
        return torch.cat(out)
    vals = {k: (preds(zva, k) == yva).float().mean().item() for k in ks}
    k = max(vals, key=vals.get)
    return {"k": k, "val_acc": vals[k], "test_acc": (preds(zte, k) == yte).float().mean().item()}


def main():
    cfg = load_config()
    data = load_data(cfg)
    dev = device()
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    out = {"a": {}, "b": {}, "c": {}}

    # ---------------- (a) bottleneck sweep
    aes = {}
    for d in cfg["ae_bottlenecks"]:
        set_seed(cfg["seed"])
        ae = ConvAE(d).to(dev)
        h = train_ae(ae, data, cfg, log=f"AE-{d}")
        save_json(h, "results", "q3", f"ae{d}_hist.json")
        torch.save(ae.state_dict(), path("results", "ckpt", f"q3_ae{d}.pt"))
        aes[d] = ae
        out["a"][str(d)] = {"params": n_params(ae), "val_mse": h["best_val_mse"],
                            "best_epoch": h["best_epoch"],
                            "compression": 3072 / d}
    sel = min(aes, key=lambda d: out["a"][str(d)]["val_mse"])
    out["a"]["selected"] = sel
    out["a"]["selected_test_mse"] = mse(recon(aes[sel], xte), xte)
    print("selected AE", sel, out["a"], flush=True)

    # ---------------- (b) denoising AE with same architecture as the selected AE
    sigma = cfg["noise_sigma"]
    set_seed(cfg["seed"])
    dae = ConvAE(sel).to(dev)
    h = train_ae(dae, data, cfg, sigma=sigma, log=f"DAE-{sel}")
    save_json(h, "results", "q3", "dae_hist.json")
    torch.save(dae.state_dict(), path("results", "ckpt", f"q3_dae{sel}.pt"))
    clf = build("CNN", data)
    clf.load_state_dict(torch.load(path("results", "ckpt", "q1a_CNN.pt")))
    g = torch.Generator().manual_seed(cfg["seed"] + 2024)
    xte_noisy = add_noise(xte, sigma, g)  # one fixed corrupted copy of the test set
    conds = {"clean": xte, "noisy": xte_noisy,
             "AE(clean)": recon(aes[sel], xte), "AE(noisy)": recon(aes[sel], xte_noisy),
             "DAE(noisy)": recon(dae, xte_noisy), "DAE(clean)": recon(dae, xte)}
    for k, x in conds.items():
        p = F.softmax(predict(clf, x), 1)
        out["b"][k] = {"cnn_test_acc": (p.argmax(1) == yte).float().mean().item(),
                       "mean_conf": p.max(1).values.mean().item(),
                       "mse_to_clean": mse(x, xte),
                       "per_class_acc": [((p.argmax(1) == yte) & (yte == c)).sum().item() /
                                         (yte == c).sum().item() for c in range(10)]}
        print(k, out["b"][k]["cnn_test_acc"], out["b"][k]["mse_to_clean"], flush=True)
    out["b"]["sigma"] = sigma
    out["b"]["dae_val_mse"] = h["best_val_mse"]
    np.savez_compressed(path("results", "q3", "recon_examples.npz"),
                        **{k: v[:16].cpu().numpy() for k, v in conds.items()})

    # ---------------- (c) class information in frozen bottleneck codes
    reps = {f"AE-{d}": aes[d] for d in cfg["ae_bottlenecks"]}
    reps[f"DAE-{sel}"] = dae
    raw = {"train": xtr.flatten(1), "val": xva.flatten(1), "test": xte.flatten(1)}
    out["c"]["raw_pixels"] = {"dim": 3072, "linear": linear_probe(raw["train"], ytr, raw["val"], yva,
                                                                 raw["test"], yte),
                              "knn": knn_probe(raw["train"], ytr, raw["val"], yva, raw["test"], yte)}
    for name, ae in reps.items():
        z = {s: encode(ae, data[s][0]) for s in ["train", "val", "test"]}
        np.savez_compressed(path("results", "q3", f"codes_{name}.npz"),
                            **{s: v.cpu().numpy() for s, v in z.items()})
        out["c"][name] = {"dim": ae.d, "test_mse": mse(recon(ae, xte), xte),
                          "linear": linear_probe(z["train"], ytr, z["val"], yva, z["test"], yte),
                          "knn": knn_probe(z["train"], ytr, z["val"], yva, z["test"], yte)}
        print(name, out["c"][name], flush=True)
    save_json(out, "results", "q3", "summary.json")


if __name__ == "__main__":
    main()
