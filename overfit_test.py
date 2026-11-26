import torch
import torch.nn.functional as F
from data import get_loaders
from models import ResNetCIFAR

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, _ = get_loaders(batch_size=128, num_workers=2)
images, labels = next(iter(train_loader))          # ONE batch, reused every step
images, labels = images.to(device), labels.to(device)

model = ResNetCIFAR(depth=20, residual=True).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

model.train()
for step in range(1, 301):
    loss = F.cross_entropy(model(images), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 1 or step % 50 == 0:
        print(f"step {step:3d} | loss {loss.item():.4f}")