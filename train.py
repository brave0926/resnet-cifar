import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import get_loaders
from models import ResNetCIFAR


def set_seed(seed):
    # same random starting point every time → reproducible runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, use_amp):
    # forward only: no backward, no update
    model.eval()
    total_loss, wrong, n = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                scores = model(images)
                loss = F.cross_entropy(scores, labels, reduction="sum")
            total_loss += loss.item()
            wrong += (scores.argmax(dim=1) != labels).sum().item()
            n += labels.size(0)
    return total_loss / n, 100.0 * wrong / n          # avg loss, error %


def main():
    # ---------- settings from the command line ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--plain", action="store_true", help="no shortcuts (default: ResNet)")
    parser.add_argument("--epochs", type=int, default=164)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="runs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"                          # mixed precision only on GPU
    residual = not args.plain
    name = f"{'resnet' if residual else 'plain'}{args.depth}"
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    # save the settings next to the results
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"device": device, "name": name}, f, indent=2)

    # ---------- data, model, optimizer ----------
    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    model = ResNetCIFAR(depth=args.depth, residual=residual).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.wd)
    # paper: lr ÷10 at 50% and 75% of training (32k and 48k of 64k iterations)
    milestones = [round(args.epochs * 0.5), round(args.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    params = sum(p.numel() for p in model.parameters())
    print(f"{name} | params {params:,} | device {device} | AMP {use_amp} | lr drops at epochs {milestones}")

    # ---------- training ----------
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_err", "test_loss", "test_err", "time_s"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            total_loss, wrong, n = 0.0, 0, 0

            for images, labels in train_loader:            # one loop = one circle
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    scores = model(images)                         # 1. forward
                    loss = F.cross_entropy(scores, labels)         # 2. loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()                                    # 3. backward
                optimizer.step()                                   # 4. update

                total_loss += loss.item() * labels.size(0)
                wrong += (scores.argmax(dim=1) != labels).sum().item()
                n += labels.size(0)

            scheduler.step()                                       # lr schedule, once per epoch
            train_loss, train_err = total_loss / n, 100.0 * wrong / n
            test_loss, test_err = evaluate(model, test_loader, device, use_amp)
            dt = time.time() - t0

            writer.writerow([epoch, lr, f"{train_loss:.4f}", f"{train_err:.2f}",
                             f"{test_loss:.4f}", f"{test_err:.2f}", f"{dt:.1f}"])
            f.flush()                                              # write to disk now
            print(f"epoch {epoch:3d}/{args.epochs} | lr {lr:.4f} | "
                  f"train loss {train_loss:.3f} err {train_err:5.2f}% | "
                  f"test loss {test_loss:.3f} err {test_err:5.2f}% | {dt:.0f}s")

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print(f"Done. Final test error: {test_err:.2f}%  → results in {out_dir}/")


if __name__ == "__main__":
    main()