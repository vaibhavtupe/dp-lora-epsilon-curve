"""Assemble final result files from the compact run table pulled off the pod.

  python assemble_results.py results_final/all_runs.csv results_final/

Writes:
  final_main.jsonl     baseline (lr 5e-4) + DP main curve at lr 2e-3 (3 seeds) + clip sweep at lr 2e-3
  alt_lr5e-3.jsonl     DP main curve at lr 5e-3 (3 seeds)  -> LR-sensitivity figure
  clip_lr5e-3.jsonl    clip sweep at lr 5e-3 (for the record)
  pilots.csv           1-epoch pilot runs
"""
import csv, json, sys, os

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
f = lambda x: None if x in ("", None) else float(x)


def rec(r, tag):
    return {"tag": tag, "private": r["private"] == "1", "lr": f(r["lr"]), "sigma": f(r["sigma"]),
            "clip": f(r["clip"]), "seed": int(r["seed"]), "acc": f(r["acc"]), "eps": f(r["eps"]),
            "wall_s": None if r["file"] == "main_dp_lr2e-3" else f(r["wall_s"]),  # ran concurrently with pilot2 -> timing not comparable
            "peak_mem_gb": f(r["peak_mem_gb"]), "delta": 1e-5, "epochs": int(r["epochs"]), "source": r["file"]}


main, alt, clip5 = [], [], []
for r in rows:
    full = r["epochs"] == "3"
    if not full:
        continue
    lr, priv, tag = float(r["lr"]), r["private"] == "1", r["tag"]
    if not priv:
        main.append(rec(r, "main"))
    elif tag in ("clip_sweep",):
        clip5.append(rec(r, "clip_sweep"))
    elif tag == "clip2e-3":
        main.append(rec(r, "clip_sweep"))
    elif abs(lr - 2e-3) < 1e-9:
        main.append(rec(r, "main"))
    elif abs(lr - 5e-3) < 1e-9:
        alt.append(rec(r, "main"))

os.makedirs(out, exist_ok=True)
for name, rs in [("final_main.jsonl", main), ("alt_lr5e-3.jsonl", alt), ("clip_lr5e-3.jsonl", clip5)]:
    with open(os.path.join(out, name), "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in rs)
    print(name, len(rs))
with open(os.path.join(out, "pilots.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader()
    w.writerows([r for r in rows if r["epochs"] != "3"])
total_s = sum(float(r["wall_s"]) for r in rows)
print(f"sum of run wall-clock: {total_s/3600:.2f} h over {len(rows)} runs")
