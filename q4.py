"""Q4: SVM C/kernel sweep on one fixed representation, difficult-pair analysis, clustering."""
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASSES, ROOT, get_split, load_config, load_json, path, save_json
import os
import pickle


def choose_representation():
    """Rule fixed before any SVM is fitted: the Q3 frozen code with the highest *validation*
    linear-probe accuracy."""
    q3 = load_json("results", "q3", "summary.json")["c"]
    cands = {k: v["linear"]["val_acc"] for k, v in q3.items() if k != "raw_pixels"}
    return max(cands, key=cands.get), cands


def fit_one(kernel, C, Xs, ys, Xv, yv):
    kw = dict(kernel=kernel, C=C, gamma="scale", cache_size=4000, max_iter=2_000_000)
    if kernel == "poly":
        kw.update(degree=3, coef0=1.0)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        svm = SVC(**kw).fit(Xs, ys)
        converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return {"kernel": kernel, "C": C, "train_acc": float((svm.predict(Xs) == ys).mean()),
            "val_acc": float((svm.predict(Xv) == yv).mean()),
            "sv_frac": float(svm.n_support_.sum() / len(ys)),
            "n_sv_per_class": svm.n_support_.tolist(), "fit_s": time.time() - t0,
            "converged": converged}, svm


def pair_column(i, j):
    """Column of the OvO decision function for classes i<j (sklearn ordering)."""
    return [(a, b) for a in range(10) for b in range(a + 1, 10)].index((i, j))


def pair_svs(svm, ys_sub, i, j):
    """Indices (into the SVM training subset) of support vectors of the (i, j) binary machine."""
    sv_idx, sv_lab = svm.support_, ys_sub[svm.support_]
    res = []
    for row, (s, lab) in enumerate(zip(sv_idx, sv_lab)):
        if lab == i:
            coef = svm.dual_coef_[j - 1, row]
        elif lab == j:
            coef = svm.dual_coef_[i, row]
        else:
            continue
        if coef != 0:
            res.append((s, abs(coef)))
    return res


def main():
    cfg = load_config()
    rep, cands = choose_representation()
    print("representation:", rep, cands, flush=True)
    z = np.load(path("results", "q3", f"codes_{rep}.npz"))
    import torchvision
    tr = torchvision.datasets.CIFAR10(os.path.join(ROOT, "data"), train=True)
    te = torchvision.datasets.CIFAR10(os.path.join(ROOT, "data"), train=False)
    tri, vai = get_split(cfg)
    ytr_all, yte = np.array(tr.targets)[tri], np.array(te.targets)
    yva = np.array(tr.targets)[vai]

    # Fixed stratified SVM subset, chosen from the seed before any model selection.
    rng = np.random.RandomState(cfg["seed"] % (2 ** 32) + 4)
    sub = np.sort(np.concatenate([rng.choice(np.where(ytr_all == c)[0], cfg["svm_subset_per_class"],
                                             replace=False) for c in range(10)]))
    np.save(path("splits", f"svm_subset_seed{cfg['seed']}.npy"), sub)
    scaler = StandardScaler().fit(z["train"][sub])
    Xs, Xv, Xt = (scaler.transform(a) for a in (z["train"][sub], z["val"], z["test"]))
    ys = ytr_all[sub]

    # ---------------- (a) sweep
    jobs = [(k, C) for k in cfg["svm_kernels"] for C in cfg["svm_C_grid"]]
    res = Parallel(n_jobs=min(15, len(jobs)), verbose=5)(
        delayed(fit_one)(k, C, Xs, ys, Xv, yv=yva) for k, C in jobs)
    sweep = [r for r, _ in res]
    for r in sweep:
        print(r, flush=True)
    best_i = max(range(len(sweep)), key=lambda i: (sweep[i]["val_acc"], -sweep[i]["sv_frac"]))
    svm = res[best_i][1]
    sel = sweep[best_i]
    pred = svm.predict(Xt)
    test_acc = float((pred == yte).mean())
    conf = np.bincount(yte * 10 + pred, minlength=100).reshape(10, 10)
    print("selected", sel["kernel"], sel["C"], "test", test_acc, flush=True)
    with open(path("results", "ckpt", "q4_svm.pkl"), "wb") as f:
        pickle.dump({"svm": svm, "scaler": scaler}, f)

    # ---------------- (b) difficult pairs
    sym = conf + conf.T
    np.fill_diagonal(sym, 0)
    pairs = sorted([(sym[i, j], i, j) for i in range(10) for j in range(i + 1, 10)], reverse=True)
    svm.decision_function_shape = "ovo"
    dte, dtr = svm.decision_function(Xt), svm.decision_function(Xs)
    # Determine sign convention empirically: positive score -> class i or j?
    c01 = pair_column(0, 1)
    m01 = np.isin(ys, [0, 1])
    pos_is_i = np.mean((dtr[m01, c01] > 0) == (ys[m01] == 0)) > 0.5
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=10).fit(Xs)

    def pair_stats(i, j):
        col = pair_column(i, j)
        s_te = dte[:, col] if pos_is_i else -dte[:, col]   # >0 means "class i"
        s_tr = dtr[:, col] if pos_is_i else -dtr[:, col]
        mt = np.isin(yte, [i, j])
        pair_acc = float(np.mean((s_te[mt] > 0) == (yte[mt] == i)))
        err = mt & np.isin(pred, [i, j]) & (pred != yte)
        svs = pair_svs(svm, ys, i, j)
        sv_ids = np.array([s for s, _ in svs])
        alphas = np.array([a for _, a in svs])
        # geometry: Fisher ratio along the SVM score, and neighbourhood mixing in the representation
        a, b = s_te[yte == i], s_te[yte == j]
        fisher = float((a.mean() - b.mean()) ** 2 / (a.var() + b.var()))
        nb = nn_model.kneighbors(Xt[mt], return_distance=False)
        own = np.where(yte[mt] == i, i, j)
        other = np.where(yte[mt] == i, j, i)
        mix = float(np.mean((ys[nb] == other[:, None]).sum(1) /
                            np.maximum(np.isin(ys[nb], [i, j]).sum(1), 1)))
        near = np.where(mt)[0][np.argsort(np.abs(s_te[mt]))[:8]]
        bound = sv_ids[alphas >= svm.C * 0.999]
        viol = bound[np.argsort(np.where(ys[bound] == i, s_tr[bound], -s_tr[bound]))[:8]] \
            if len(bound) else np.array([], int)
        margin = sv_ids[np.argsort(np.abs(s_tr[sv_ids]))[:8]]
        return {"pair": [CLASSES[i], CLASSES[j]], "i": i, "j": j,
                "confusions": int(sym[i, j]), "i_as_j": int(conf[i, j]), "j_as_i": int(conf[j, i]),
                "pairwise_test_acc": pair_acc, "n_pair_svs": int(len(sv_ids)),
                "pair_sv_frac": float(len(sv_ids) / np.isin(ys, [i, j]).sum()),
                "frac_bound_svs": float(len(bound) / max(len(sv_ids), 1)),
                "frac_errors_within_margin": float(np.mean(np.abs(s_te[err]) < 1)) if err.any() else None,
                "frac_pair_test_within_margin": float(np.mean(np.abs(s_te[mt]) < 1)),
                "fisher_ratio": fisher, "knn_mixing": mix,
                "near_test": [{"idx": int(k), "label": CLASSES[yte[k]], "pred": CLASSES[pred[k]],
                               "score": float(s_te[k])} for k in near],
                "violating_svs": [{"idx": int(tri[sub[k]]), "label": CLASSES[ys[k]],
                                   "score": float(s_tr[k])} for k in viol],
                "margin_svs": [{"idx": int(tri[sub[k]]), "label": CLASSES[ys[k]],
                                "score": float(s_tr[k])} for k in margin]}

    hard = [pair_stats(i, j) for _, i, j in pairs[:3]]
    easy = [pair_stats(i, j) for _, i, j in pairs[-2:]]
    all_pairs = {f"{i}-{j}": pair_stats(i, j) for _, i, j in pairs}

    # ---------------- (c) clustering: fit on train representations, select on validation
    sc_full = StandardScaler().fit(z["train"])
    pca = PCA(cfg["cluster_pca_dims"], random_state=0).fit(sc_full.transform(z["train"]))
    P = {s: pca.transform(sc_full.transform(z[s])) for s in ["train", "val", "test"]}
    csweep, models = [], {}
    for method in ["kmeans", "gmm"]:
        for k in cfg["cluster_k_grid"]:
            if method == "kmeans":
                m = KMeans(k, n_init=10, random_state=cfg["seed"] % (2 ** 32)).fit(P["train"])
                extra = {"val_inertia_per_pt": float(-m.score(P["val"]) / len(P["val"]))}
            else:
                m = GaussianMixture(k, covariance_type="diag", n_init=2,
                                    random_state=cfg["seed"] % (2 ** 32)).fit(P["train"])
                extra = {"val_loglik": float(m.score(P["val"])), "train_bic": float(m.bic(P["train"]))}
            cv = m.predict(P["val"])
            r = {"method": method, "k": k, "val_ari": float(adjusted_rand_score(yva, cv)),
                 "val_nmi": float(normalized_mutual_info_score(yva, cv)), **extra}
            csweep.append(r); models[(method, k)] = m
            print(r, flush=True)
    cbest = max(csweep, key=lambda r: r["val_ari"])
    cm = models[(cbest["method"], cbest["k"])]
    ct = cm.predict(P["test"])                 # assignment without labels
    K = cbest["k"]
    cont = np.zeros((K, 10), int)
    np.add.at(cont, (ct, yte), 1)
    clus = {"selected": cbest, "test_ari": float(adjusted_rand_score(yte, ct)),
            "test_nmi": float(normalized_mutual_info_score(yte, ct)),
            "svm_test_ari": float(adjusted_rand_score(yte, pred)),
            "svm_test_nmi": float(normalized_mutual_info_score(yte, pred)),
            "contingency": cont.tolist(), "sweep": csweep}

    # Classification vs clustering per pair: SVM pairwise accuracy vs clustering pairwise
    # separability (best majority mapping of clusters within the pair, evaluated with test labels
    # only after assignment).
    per_pair = []
    for _, i, j in pairs:
        mt = np.isin(yte, [i, j])
        ci, cj = cont[:, i], cont[:, j]
        clus_acc = float(np.maximum(ci, cj).sum() / (ci.sum() + cj.sum()))
        per_pair.append({"pair": [CLASSES[i], CLASSES[j]],
                         "svm_pairwise_acc": all_pairs[f"{i}-{j}"]["pairwise_test_acc"],
                         "cluster_pairwise_acc": clus_acc,
                         "cluster_pair_ari": float(adjusted_rand_score(yte[mt], ct[mt]))})
    per_class = []
    for c in range(10):
        dom = int(cont[:, c].argmax())
        per_class.append({"class": CLASSES[c], "svm_recall": float(conf[c, c] / conf[c].sum()),
                          "svm_precision": float(conf[c, c] / conf[:, c].sum()),
                          "dominant_cluster": dom,
                          "frac_in_dominant_cluster": float(cont[dom, c] / cont[:, c].sum()),
                          "dominant_cluster_purity": float(cont[dom, c] / cont[dom].sum()),
                          "n_clusters_for_80pct": int(np.searchsorted(
                              np.cumsum(np.sort(cont[:, c])[::-1]) / cont[:, c].sum(), 0.8) + 1)})
    clus["per_pair"] = per_pair
    clus["per_class"] = per_class

    save_json({"representation": rep, "rep_candidates_val_linear": cands,
               "subset_size": int(len(sub)), "sweep": sweep, "selected": sel,
               "test_acc": test_acc, "test_confusion": conf.tolist(),
               "per_class_test_acc": (np.diag(conf) / conf.sum(1)).tolist(),
               "pos_score_is_first_class": bool(pos_is_i),
               "hard_pairs": hard, "easy_pairs": easy, "clustering": clus},
              "results", "q4", "summary.json")


if __name__ == "__main__":
    main()
