#!/usr/bin/env bash
# End-to-end sweep for the poster. Unattended:
#   0) pilot: 1-epoch LR check for baseline and DP (eps~2)  -> picks BASE_LR / DP_LR automatically
#   1) main curve: non-private baseline + 5 noise multipliers x 3 seeds
#   2) clipping-norm sweep at eps~2
#   3) friction demo + figures
#
# Noise multipliers were calibrated (Opacus RDP accountant) for
# roberta-base / SST-2 (N=67,349), batch 1024, 3 epochs, delta=1e-5:
#   sigma 0.576 -> eps ~8 | 0.736 -> ~4 | 0.954 -> ~2 | 1.287 -> ~1 | 1.918 -> ~0.5
# If you change model/task/batch/epochs, re-run calibrate_sigma.py.
#
# Override: BASE_LR=... DP_LR=... SKIP_PILOT=1 SEEDS="0 1 2" ./run_sweep.sh
set -uo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-results/results.jsonl}
SEEDS=${SEEDS:-"0 1 2"}
SIGMAS="0.576 0.736 0.954 1.287 1.918"
mkdir -p results logs figures
status() { echo "$(date -u +%FT%TZ) $*" | tee -a results/STATUS.txt; }

run() {  # run <lr> <args...> ; skips configs already present in $OUT (resume-safe)
  local lr=$1; shift
  local key="$*"
  if [[ -f results/done.txt ]] && grep -qxF -- "$key" results/done.txt; then echo "skip: $key"; return; fi
  status "START $key (lr=$lr)"
  if python dp_lora.py --out "$OUT" --lr "$lr" "$@" ${EXTRA:-} >> logs/sweep.log 2>&1; then
    echo "$key" >> results/done.txt; status "DONE  $key"
  else
    status "FAIL  $key  (see logs/sweep.log)"
  fi
}

# ---- preflight: tiny real-data runs to fail fast (dataset, model download, Opacus wiring)
if [[ ! -f results/preflight.ok ]]; then
  status "PREFLIGHT start"
  python dp_lora.py --out results/preflight.jsonl --tag preflight --sigma 0   --epochs 1 --max_train 4096 --batch 256 ${EXTRA:-} >> logs/preflight.log 2>&1 \
   && python dp_lora.py --out results/preflight.jsonl --tag preflight --sigma 1.0 --epochs 1 --max_train 4096 --batch 256 ${EXTRA:-} >> logs/preflight.log 2>&1 \
   && touch results/preflight.ok && status "PREFLIGHT ok" \
   || { status "PREFLIGHT FAILED - see logs/preflight.log"; exit 1; }
fi

# ---- 0) pilot
if [[ "${SKIP_PILOT:-0}" != "1" && ( -z "${BASE_LR:-}" || -z "${DP_LR:-}" ) ]]; then
  status "PILOT start"
  for lr in 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2; do
    for sg in 0 0.954; do
      key="pilot lr=$lr sigma=$sg"
      grep -qxF -- "$key" results/done.txt 2>/dev/null && continue
      python dp_lora.py --out results/pilot.jsonl --tag pilot --lr $lr --sigma $sg --epochs 1 --seed 0 ${EXTRA:-} >> logs/pilot.log 2>&1 \
        && echo "$key" >> results/done.txt
    done
  done
  read BASE_LR DP_LR < <(python - <<'EOF'
import json
rs=[json.loads(l) for l in open("results/pilot.jsonl")]
best=lambda priv: max((r for r in rs if r["private"]==priv), key=lambda r: r["acc"])["lr"]
print(best(False), best(True))
EOF
)
  status "PILOT picked BASE_LR=$BASE_LR DP_LR=$DP_LR"
fi
BASE_LR=${BASE_LR:-5e-4}; DP_LR=${DP_LR:-5e-4}

# ---- 1) main curve
for s in $SEEDS; do
  run "$BASE_LR" --tag main --sigma 0 --seed "$s"
  for sg in $SIGMAS; do run "$DP_LR" --tag main --sigma "$sg" --clip 1.0 --seed "$s"; done
done

# ---- 2) clipping sensitivity at eps~2 (C=1.0 comes from the main curve)
for c in 0.1 0.3 3.0 10.0; do run "$DP_LR" --tag clip_sweep --sigma 0.954 --clip "$c" --seed 0; done

# ---- 3) friction demo + figures
python friction_demo.py > results/friction_demo.txt 2>&1
python plot.py --inp "$OUT" --outdir figures > results/summary.txt 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv > results/gpu.txt 2>&1
status "ALL DONE"
touch results/ALL_DONE
