import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


class CNN4BN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
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

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    ckpt_path = "checkpoints/cifar10_simplecnn.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = CNN4BN(num_classes=10).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Loaded:", ckpt_path)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    overall_acc = (y_pred == y_true).mean()
    print(f"Overall test accuracy: {overall_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    plt.figure(figsize=(9, 8))
    plt.imshow(cm)
    plt.title("CIFAR-10 Confusion Matrix (CNN4BN)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), classes, rotation=45, ha="right")
    plt.yticks(range(10), classes)

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            plt.text(j, i, str(val), ha="center", va="center",
                     fontsize=8,
                     color=("white" if val > max_val * 0.6 else "black"))

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    print("Saved: confusion_matrix.png")

    print("\nPer-class accuracy:")
    per_class_acc = {}
    for cls_id, cls_name in enumerate(classes):
        mask = (y_true == cls_id)
        acc = (y_pred[mask] == y_true[mask]).mean() if mask.any() else float("nan")
        per_class_acc[cls_name] = acc
        print(f"{cls_name:>10s}: {acc:.4f}")

    with open("per_class_accuracy.csv", "w", encoding="utf-8") as f:
        f.write("class,accuracy\n")
        for cls_name in classes:
            f.write(f"{cls_name},{per_class_acc[cls_name]:.6f}\n")
    print("\nSaved: per_class_accuracy.csv")

    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved: classification_report.txt")


if __name__ == "__main__":
    main()
