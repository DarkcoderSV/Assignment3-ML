"""Robustness check for Q1(a): MLP-M and CNN both selected the largest LR in the grid (3e-3), so
we also train them at 1e-2. Validation accuracy only (the test set is not touched)."""
from common import load_config, load_data, save_json, set_seed, train_classifier
from q1 import build

cfg = load_config()
data = load_data(cfg)
out = {}
for name in ["MLP-M", "CNN"]:
    set_seed(cfg["seed"])
    m = build(name, data)
    h = train_classifier(m, data, 0.01, cfg["clf_epochs"], cfg["clf_weight_decay"], cfg["batch_size"],
                         seed=cfg["seed"], log=f"{name} lr=0.01")
    out[name] = {"lr": 0.01, "best_val_acc": h["best_val_acc"], "best_epoch": h["best_epoch"]}
    print(name, out[name], flush=True)
save_json(out, "results", "q1a", "lr_edge_check.json")
