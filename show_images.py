import matplotlib.pyplot as plt
from torchvision import datasets

data = datasets.CIFAR10("./data", train=True, download=True)  # no transforms → original images

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i, ax in enumerate(axes.flat):
    image, label = data[i]
    ax.imshow(image)
    ax.set_title(data.classes[label])
    ax.axis("off")
plt.tight_layout()
plt.savefig("cifar_samples.png")
print("Saved cifar_samples.png")