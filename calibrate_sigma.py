"""Find the noise multiplier that lands on each target epsilon (Opacus RDP accountant).

  python calibrate_sigma.py --n 67349 --batch 1024 --epochs 3 --delta 1e-5
"""
import argparse, math
from opacus.accountants.utils import get_noise_multiplier

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=67349)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--delta", type=float, default=1e-5)
p.add_argument("--targets", default="8,4,2,1,0.5")
a = p.parse_args()
q = a.batch / a.n
steps = math.ceil(a.epochs * a.n / a.batch)
print(f"q={q:.4f}  steps={steps}  delta={a.delta}")
for t in map(float, a.targets.split(",")):
    s = get_noise_multiplier(target_epsilon=t, target_delta=a.delta, sample_rate=q, steps=steps, accountant="rdp")
    print(f"  target eps {t:>4}: sigma = {s:.3f}")
