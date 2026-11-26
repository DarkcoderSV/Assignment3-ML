"""Writes the long Q4 LaTeX table bodies (sweep, clustering sweep, contingency) from results/."""
from common import load_json, path

s = load_json("results", "q4", "summary.json")
rows = []
for r in s["sweep"]:
    sel = r["kernel"] == s["selected"]["kernel"] and r["C"] == s["selected"]["C"]
    conv = "" if r["converged"] else "$^\\ddagger$"
    line = f"{r['kernel']} & {r['C']:g} & {r['train_acc']:.3f} & {r['val_acc']:.3f} & {r['sv_frac']:.3f}{conv}\\\\"
    rows.append(("\\rowcolor{yellow!25}" if sel else "") + line)
open(path("report", "q4_sweep_rows.tex"), "w").write("\n".join(rows))
c = s["clustering"]
open(path("report", "q4_cluster_rows.tex"), "w").write("\n".join(
    f"{r['method']} & {r['k']} & {r['val_ari']:.3f} & {r['val_nmi']:.3f}\\\\" for r in c["sweep"]))
lines = []
for k, row in enumerate(c["contingency"]):
    m = max(row)
    lines.append(f"c{k} & " + " & ".join((f"\\textbf{{{v}}}" if v == m else str(v)) for v in row)
                 + f" & {sum(row)}\\\\")
open(path("report", "q4_contingency_rows.tex"), "w").write("\n".join(lines))
