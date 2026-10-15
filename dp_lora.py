"""
DP-SGD + LoRA fine-tuning with Opacus — one run per invocation.

Poster: "What Privacy Actually Costs: An Accuracy-vs-Epsilon Curve for LoRA Fine-Tuning"
PyTorch Conference NA 2026 — Shrinath Thube (IBM) & Vaibhav Tupe (Crusoe)

Each run appends one JSON line to --out with: config, final eval accuracy,
per-epoch accuracy, epsilon (RDP accountant, fixed delta), wall time,
peak GPU memory, trainable-parameter count and throughput.

Examples
  # non-private baseline
  python dp_lora.py --sigma 0 --seed 0
  # DP run (sigma=0.954 -> eps ~= 2 at the default config)
  python dp_lora.py --sigma 0.954 --clip 1.0 --seed 0
  # CPU smoke test (tiny random model, synthetic data, no downloads)
  python dp_lora.py --smoke --sigma 1.0
"""
import argparse, json, math, os, random, time, warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
warnings.filterwarnings("ignore", message=".*Optimal order is the largest alpha.*")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--task", default="sst2", help="GLUE task; sst2 is the default")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_targets", default="query,value")
    p.add_argument("--sigma", type=float, default=0.0, help="noise multiplier; 0 = non-private baseline")
    p.add_argument("--clip", type=float, default=1.0, help="per-sample clipping norm C")
    p.add_argument("--batch", type=int, default=1024, help="logical (expected) batch size")
    p.add_argument("--phys_batch", type=int, default=128, help="physical micro-batch (memory knob only)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="main", help="free-text label, e.g. main / clip_sweep")
    p.add_argument("--out", default="results/results.jsonl")
    p.add_argument("--max_train", type=int, default=0, help="subsample train set (preflight checks)")
    p.add_argument("--smoke", action="store_true", help="tiny random model + synthetic data on CPU")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- data
def load_data(args, tok):
    if args.smoke:
        g = torch.Generator().manual_seed(0)
        def synth(n):
            y = torch.randint(0, 2, (n,), generator=g)
            ids = torch.randint(5, 260, (n, args.max_len), generator=g) + 240 * y[:, None]  # class shifts vocab range
            ids[:, 0] = 0
            return TensorDataset(ids, torch.ones_like(ids), y)
        return synth(2048), synth(256)

    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", args.task)
    key = {"sst2": ("sentence", None), "qnli": ("question", "sentence"),
           "mnli": ("premise", "hypothesis"), "qqp": ("question1", "question2")}[args.task]

    def enc(split):
        a = list(ds[split][key[0]])
        b = list(ds[split][key[1]]) if key[1] else None
        t = tok(a, b, truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
        return TensorDataset(t["input_ids"], t["attention_mask"], torch.tensor(list(ds[split]["label"])))

    val = "validation_matched" if args.task == "mnli" else "validation"
    return enc("train"), enc(val)


# ----------------------------------------------------------------------------- model
def build_model(args, n_labels=2):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, RobertaConfig, RobertaForSequenceClassification

    if args.smoke:
        cfg = RobertaConfig(vocab_size=520, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=128, max_position_embeddings=args.max_len + 4, num_labels=n_labels)
        base = RobertaForSequenceClassification(cfg)
    else:
        base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=n_labels)

    lcfg = LoraConfig(task_type="SEQ_CLS", r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                      target_modules=args.lora_targets.split(","))
    model = get_peft_model(base, lcfg)
    return model


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        correct += (logits.argmax(-1) == y).sum().item(); total += y.numel()
    model.train()
    return correct / max(total, 1)


# ----------------------------------------------------------------------------- main
def main():
    args = parse()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    if args.smoke:
        args.batch, args.phys_batch = min(args.batch, 256), min(args.phys_batch, 64)

    tok = None
    if not args.smoke:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
    train_ds, val_ds = load_data(args, tok)
    if args.max_train and args.max_train < len(train_ds):
        g = torch.Generator().manual_seed(args.seed)
        train_ds = torch.utils.data.Subset(train_ds, torch.randperm(len(train_ds), generator=g)[: args.max_train].tolist())
    model = build_model(args).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    private = args.sigma > 0
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    val_loader = DataLoader(val_ds, batch_size=256)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)

    engine = None
    if private:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        errs = ModuleValidator.validate(model, strict=False)
        if errs:  # transformers use LayerNorm, so this is normally empty — see friction_demo.py for BatchNorm
            raise SystemExit(f"Opacus-incompatible modules: {errs}")
        engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        model, opt, train_loader = engine.make_private(
            module=model, optimizer=opt, data_loader=train_loader,
            noise_multiplier=args.sigma, max_grad_norm=args.clip, poisson_sampling=True)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loss_fn = torch.nn.CrossEntropyLoss()
    per_epoch, n_seen = [], 0
    t0 = time.time()
    model.train()

    for ep in range(args.epochs):
        if private:
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            ctx = BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=args.phys_batch, optimizer=opt)
            loader = ctx.__enter__()
        else:
            loader = train_loader
        for ids, mask, y in loader:
            if y.numel() == 0:  # Poisson sampling can yield empty batches
                continue
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            if private:
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss_fn(logits, y).backward()
                opt.step(); opt.zero_grad(set_to_none=True)  # BMM only takes a real step at logical-batch boundaries
            else:  # grad accumulation to match the logical batch
                opt.zero_grad(set_to_none=True)
                for s in range(0, y.numel(), args.phys_batch):
                    sl = slice(s, s + args.phys_batch)
                    logits = model(input_ids=ids[sl], attention_mask=mask[sl]).logits
                    (loss_fn(logits, y[sl]) * y[sl].numel() / y.numel()).backward()
                opt.step()
            n_seen += y.numel()
        if private:
            ctx.__exit__(None, None, None)
        acc = evaluate(model, val_loader, device)
        eps_now = engine.get_epsilon(args.delta) if private else float("inf")
        per_epoch.append({"epoch": ep + 1, "acc": acc, "eps": eps_now})
        print(f"[ep {ep+1}] acc={acc:.4f} eps={eps_now:.3f}", flush=True)

    wall = time.time() - t0
    rec = {
        "tag": args.tag, "task": args.task, "model": "smoke-tiny" if args.smoke else args.model,
        "private": private, "sigma": args.sigma, "clip": args.clip if private else None,
        "batch": args.batch, "epochs": args.epochs, "lr": args.lr, "delta": args.delta, "seed": args.seed,
        "lora_r": args.lora_r, "lora_targets": args.lora_targets,
        "acc": per_epoch[-1]["acc"], "eps": per_epoch[-1]["eps"] if private else None,
        "per_epoch": per_epoch, "wall_s": round(wall, 1),
        "samples_per_s": round(n_seen / wall, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
        "trainable_params": trainable, "total_params": total_params,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "versions": {"torch": torch.__version__, "opacus": __import__("opacus").__version__},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("tag", "sigma", "clip", "seed", "acc", "eps", "wall_s", "peak_mem_gb")}))


if __name__ == "__main__":
    main()
