import argparse
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from data import MEAN, STD
from models import ResNetCIFAR

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# same preparation as the test images during training
to_input = transforms.Compose([
    transforms.Resize((32, 32)),        # any photo -> 32x32 (the size the network knows)
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict(model, pil_image, device):
    x = to_input(pil_image).unsqueeze(0).to(device)   # [3,32,32] -> [1,3,32,32] (batch of 1)
    with torch.no_grad():                             # forward only, no learning
        scores = model(x)                             # 10 raw scores
        probs = F.softmax(scores, dim=1)[0]           # scores -> probabilities (sum = 100%)
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs_pilot/resnet20/model.pt")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--image", default=None, help="path to your own image (jpg/png)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. rebuild the same network and load the trained weights into it
    model = ResNetCIFAR(depth=args.depth, residual=not args.plain)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()                           # eval mode: BN uses learned statistics

    if args.image:
        # 2a. YOUR OWN IMAGE -> top-3 guesses
        img = Image.open(args.image).convert("RGB")
        probs = predict(model, img, device)
        top = probs.topk(3)
        print(f"Image: {args.image}")
        for p, idx in zip(top.values, top.indices):
            print(f"  {CLASSES[idx]:10s} {p.item():6.1%}")
    else:
        # 2b. 8 RANDOM CIFAR-10 TEST IMAGES -> picture with prediction vs truth
        test_set = datasets.CIFAR10("./data", train=False, download=True)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for ax in axes.flat:
            img, label = test_set[random.randint(0, len(test_set) - 1)]
            probs = predict(model, img, device)
            pred = probs.argmax().item()
            ax.imshow(img)
            ax.set_title(f"pred: {CLASSES[pred]} ({probs[pred]:.0%})\ntrue: {CLASSES[label]}",
                         color="green" if pred == label else "red")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("predictions.png")
        print("Saved predictions.png (green = correct, red = wrong)")


if __name__ == "__main__":
    main()