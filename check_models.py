import torch
from models import ResNetCIFAR

x = torch.randn(2, 3, 32, 32)   # 2 fake CIFAR images

for depth in (20, 56):
    for residual in (False, True):
        model = ResNetCIFAR(depth=depth, residual=residual)
        params = sum(p.numel() for p in model.parameters())
        layers = sum(1 for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
        out = model(x)
        name = f"{'ResNet' if residual else 'plain'}-{depth}"
        print(f"{name:10s} | params = {params:,} ({params/1e6:.3f}M) | layers = {layers} | output = {tuple(out.shape)}")