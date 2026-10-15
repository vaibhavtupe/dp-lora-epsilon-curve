# What Privacy Actually Costs — experiment kit

PyTorch Conference NA 2026 poster (Responsible AI) — Shrinath Thube (IBM) & Vaibhav Tupe (Crusoe)

An accuracy-vs-ε curve for LoRA fine-tuning under DP-SGD (Opacus), with a non-private
baseline, a clipping-norm sensitivity check, and the compute cost of the sweep.

## Setup (defaults chosen for credibility + speed)

| Choice | Value | Why |
|---|---|---|
| Model | `roberta-base` (125M), LoRA r=8 on `query,value` + classifier head | Standard DP-NLP benchmark; comparable to published DP-LoRA results |
| Task | GLUE SST-2 (67,349 train / 872 val) | Big enough for meaningful ε, small enough to sweep fast |
| DP | Opacus, Poisson sampling, RDP accountant, δ = 1e-5 | δ < 1/N |
| Batch / epochs | 1024 logical (128 physical via `BatchMemoryManager`) / 3 epochs | 198 steps |
| Noise multipliers σ | 0.576, 0.736, 0.954, 1.287, 1.918 | Calibrated to ε ≈ 8, 4, 2, 1, 0.5 |
| Clip norm C | 1.0 (main); sweep 0.1, 0.3, 1, 3, 10 at ε≈2 | Friction point |
| Learning rate | 1-epoch pilot over 2e-4…1e-2; picked 5e-4 (non-private) and 5e-3 (DP) | DP needs ~10× the LR; too low a LR stalls at chance under heavy noise |
| Seeds | 3 for the main curve | Error bars on the poster |

## Run it (any single A100/H100/L40S; we used 1× L40S on RunPod)

```bash
pip install -r requirements.txt
python dp_lora.py --smoke --sigma 1.0          # 30-s CPU sanity check, no downloads

./run_sweep.sh            # preflight -> LR pilot (auto-picks LRs) -> main curve -> clip sweep -> figures
# or skip the pilot: BASE_LR=5e-4 DP_LR=5e-3 SKIP_PILOT=1 ./run_sweep.sh
```

Outputs: `results/results.jsonl` (one line per run), `figures/fig1_acc_vs_eps.{pdf,png}`,
`fig2_clip`, `fig3_cost`, `summary.csv|md`, `results/friction_demo.txt`.

The pilot picks the LR separately for the non-private baseline and for DP (they differ ~10×).
Progress is appended to `results/STATUS.txt`; the script is resume-safe.

## Files

- `dp_lora.py` — one run (baseline if `--sigma 0`); logs acc, ε, wall time, peak memory
- `run_sweep.sh` — pilot + main curve + clip sweep + friction demo + plots
- `calibrate_sigma.py` — re-derive σ if you change model / task / batch / epochs
- `friction_demo.py` — BatchNorm vs LayerNorm under `ModuleValidator` (+ auto-fix to GroupNorm)
- `plot.py` — poster figures (`--preview` watermarks synthetic data)

## Notes for the poster narrative

- **BatchNorm**: RoBERTa/LLMs use LayerNorm → no issue. It bites when people LoRA-tune CNN/vision
  backbones; Opacus rejects BatchNorm and `ModuleValidator.fix()` swaps in GroupNorm. Frame it that way.
- **LoRA helps DP**: per-sample gradients only for trainable params (~0.7% of RoBERTa-base with r=8),
  so the per-sample memory cost is small compared with DP full fine-tuning.
- **Compute cost**: report the multiplier vs. non-private and total GPU-hours for the whole sweep —
  the honest cost of choosing ε is a sweep, not a run.
- Report ε from the accountant actually used (RDP); PRV gives slightly tighter numbers — mention it.

## Results

Final numbers and figures are in `results/` and `figures/` (added after the run).

## Citation

Thube, S. and Tupe, V. *What Privacy Actually Costs: An Accuracy-vs-Epsilon Curve for LoRA Fine-Tuning.* Poster, PyTorch Conference North America 2026.
