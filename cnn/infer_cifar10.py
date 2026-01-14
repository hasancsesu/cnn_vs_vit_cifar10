import os
import random
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


class CNN4BN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8->4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4->2
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)

    test_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(cifar10_mean, cifar10_std),
    ])

    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_tfms
    )
    classes = test_ds.classes

    ckpt_path = "checkpoints/cifar10_simplecnn.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = CNN4BN(num_classes=10).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Loaded:", ckpt_path)

    N = 12
    indices = random.sample(range(len(test_ds)), N)

    imgs, true_labels = [], []
    for idx in indices:
        x, y = test_ds[idx]
        imgs.append(x)
        true_labels.append(y)

    batch = torch.stack(imgs).to(device)

    with torch.no_grad():
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().tolist()

    mean = torch.tensor(cifar10_mean).view(3, 1, 1)
    std = torch.tensor(cifar10_std).view(3, 1, 1)

    def denorm(img_tensor):
        x = img_tensor.cpu() * std + mean
        return x.clamp(0, 1)

    cols = 4
    rows = math.ceil(N / cols)

    plt.figure(figsize=(12, 8))
    for i in range(N):
        plt.subplot(rows, cols, i + 1)
        img = denorm(imgs[i]).permute(1, 2, 0)
        plt.imshow(img)
        t = classes[true_labels[i]]
        p = classes[preds[i]]
        plt.title(f"T: {t}\nP: {p}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    out_path = "predictions_grid.png"
    plt.savefig(out_path, dpi=160)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
