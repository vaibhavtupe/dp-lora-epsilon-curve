# What Privacy Actually Costs — experiment kit

PyTorch Conference NA 2026 (Responsible AI)

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
| Learning rate | 5e-4 (non-private), 2e-3 (DP), confirmed with full 3-epoch runs | DP wanted 4× the non-private LR. A 1-epoch pilot picked 5e-3, which was worse at every ε over 3 epochs |
| Seeds | 3 for the main curve | Error bars on the poster |

## Run it (any single A100/H100/L40S; we used 1× L40S on Crusoe Cloud)

```bash
pip install -r requirements.txt
python dp_lora.py --smoke --sigma 1.0          # 30-s CPU sanity check, no downloads

./run_sweep.sh            # preflight -> LR pilot (auto-picks LRs) -> main curve -> clip sweep -> figures
# or skip the pilot: BASE_LR=5e-4 DP_LR=2e-3 SKIP_PILOT=1 ./run_sweep.sh
```

Outputs: `results/results.jsonl` (one line per run), `figures/fig1_acc_vs_eps.{pdf,png}`,
`fig2_clip`, `fig3_cost`, `summary.csv|md`, `results/friction_demo.txt`.

The pilot picks the LR separately for the non-private baseline and for DP. Treat it as a starting point: in our runs the 1-epoch pilot over-picked the DP LR, so confirm at full training length.
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

Validation accuracy on SST-2 (872 examples), mean ± std over 3 seeds. Non-private LR 5e-4, DP LR 2e-3, C = 1.0.

| setting     |   sigma | eps   |   acc_mean |   acc_std |   gap_pts |   seeds |   wall_min |   peak_mem_gb |
|:------------|--------:|:------|-----------:|----------:|----------:|--------:|-----------:|--------------:|
| non-private |   0     | ∞     |      93.42 |      0.19 |      0    |       3 |        6.1 |          7.45 |
| DP-SGD      |   1.918 | 0.5   |      90.33 |      0.63 |      3.09 |       3 |        6.3 |          7.73 |
| DP-SGD      |   1.287 | 1.0   |      90.6  |      0.53 |      2.83 |       3 |        6.3 |          7.73 |
| DP-SGD      |   0.954 | 2.0   |      90.75 |      0.51 |      2.68 |       3 |        6.3 |          7.73 |
| DP-SGD      |   0.736 | 4.0   |      90.83 |      0.23 |      2.6  |       3 |        6.3 |          7.73 |
| DP-SGD      |   0.576 | 7.97  |      90.86 |      0.24 |      2.56 |       3 |        6.3 |          7.73 |

- **Privacy cost is small and nearly flat:** DP-LoRA held 90.3–90.9% from ε ≈ 8 down to ε ≈ 0.5, which is 2.6–3.1 points below non-private.
- **Learning rate mattered more than ε:** at DP LR 5e-3 (the 1-epoch pilot's pick) the same runs lost 1.3–3.7 extra points, and the gap grew as ε shrank (`figures/fig4_lr`, `results/alt_lr5e-3.jsonl`).
- **Clipping norm C was not sensitive:** at ε ≈ 2 and LR 2e-3, C from 0.1 to 10 stayed within 0.4 points (`figures/fig2_clip`).
- **Compute:** a DP run took 6.3 vs 6.1 min for non-private on 1× L40S (peak memory 7.7 vs 7.5 GB). The whole study was 55 runs, ≈ 5 GPU-hours.

Files: `results/all_runs.csv` (every run, including pilots), `results/final_main.jsonl` (inputs to the poster figures; rebuild with
`python plot.py --inp results/final_main.jsonl --outdir figures --no_titles --lr_extra results/alt_lr5e-3.jsonl`), `poster.pdf`.


## Citation

*What Privacy Actually Costs: An Accuracy-vs-Epsilon Curve for LoRA Fine-Tuning.* PyTorch Conference North America 2026.
