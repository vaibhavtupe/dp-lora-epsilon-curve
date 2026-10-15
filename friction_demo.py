"""
Friction point #1 from the abstract: BatchNorm incompatibility.

BatchNorm mixes statistics across the samples in a batch, so per-sample gradients
(and therefore per-sample clipping) are undefined. Opacus refuses such models.
Transformer backbones (RoBERTa, Llama, ...) use LayerNorm and pass; CNN / some
vision backbones with LoRA adapters do not. This script prints both outcomes
and the automatic fix (BatchNorm -> GroupNorm), for a poster callout.

  python friction_demo.py
"""
import torch.nn as nn
from opacus.validators import ModuleValidator
from peft import LoraConfig, get_peft_model
from transformers import RobertaConfig, RobertaForSequenceClassification


class TinyCNN(nn.Module):  # stands in for a ResNet-style backbone
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3); self.bn = nn.BatchNorm2d(16)
        self.head = nn.Linear(16, 2)
    def forward(self, x):
        return self.head(self.bn(self.conv(x)).mean((2, 3)))


def report(name, model):
    errs = ModuleValidator.validate(model, strict=False)
    print(f"\n{name}: {'OK - Opacus-compatible' if not errs else f'{len(errs)} incompatible module(s)'}")
    for e in errs:
        print("   -", str(e).splitlines()[0][:110])
    return errs


cnn = TinyCNN()
if report("CNN + BatchNorm", cnn):
    fixed = ModuleValidator.fix(cnn)
    report("CNN after ModuleValidator.fix()", fixed)
    print("   ->", [type(m).__name__ for m in fixed.modules() if "Norm" in type(m).__name__])

cfg = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
lora = get_peft_model(RobertaForSequenceClassification(cfg),
                      LoraConfig(task_type="SEQ_CLS", r=8, target_modules=["query", "value"]))
report("RoBERTa + LoRA (LayerNorm)", lora)
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"   trainable params: {trainable:,} of {sum(p.numel() for p in lora.parameters()):,}"
      " - only these get per-sample gradients")
