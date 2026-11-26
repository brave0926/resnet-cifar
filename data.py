import torch
from torchvision import datasets, transforms

# Average and spread of pixel values in CIFAR-10 (for R, G, B).
# Used to "normalize" images: center the numbers around 0.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_loaders(batch_size=128, num_workers=4, root="./data"):
    # Training images: random changes every time (augmentation, from the paper)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # add 4-pixel border, cut a random 32x32 piece
        transforms.RandomHorizontalFlip(),      # mirror left-right, 50% chance
        transforms.ToTensor(),                  # picture -> numbers in [0, 1], shape [3, 32, 32]
        transforms.Normalize(MEAN, STD),        # center numbers around 0
    ])

    # Test images: no random changes, we measure on the real images
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Load the datasets (downloads only if the files are missing)
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)

    # Loaders cut the data into batches
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick test: runs only when you do "python data.py"
    train_loader, test_loader = get_loaders()

    print("Training images:", len(train_loader.dataset))
    print("Test images:    ", len(test_loader.dataset))
    print("Classes:", train_loader.dataset.classes)

    images, labels = next(iter(train_loader))   # take ONE batch
    print("Batch images shape:", images.shape)  # expect [128, 3, 32, 32]
    print("Batch labels shape:", labels.shape)  # expect [128]
    print("First 10 labels:", labels[:10].tolist())
    print(f"Pixel mean: {images.mean():.3f}  std: {images.std():.3f}")