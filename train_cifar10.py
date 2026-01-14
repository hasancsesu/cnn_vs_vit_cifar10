print("SCRIPT STARTED")

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt


# -----------------------------
# 1) Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# 2) Data (CIFAR-10)
# -----------------------------
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std  = (0.2470, 0.2435, 0.2616)

train_tfms = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(cifar10_mean, cifar10_std),
])

test_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize(cifar10_mean, cifar10_std),
])

data_dir = "./data"

train_ds = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=train_tfms
)
test_ds = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=test_tfms
)

batch_size = 128
num_workers = 0

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

classes = train_ds.classes
print("Classes:", classes)


# -----------------------------
# 3) Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
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
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN(num_classes=10).to(device)
print(model)


# -----------------------------
# 4) Helpers
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_count += y.size(0)

    return total_loss / total_count, total_correct / total_count


# -----------------------------
# 5) Training setup
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []


# -----------------------------
# 6) Train loop
# -----------------------------
for epoch in range(1, epochs + 1):
    t0 = time.time()
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_count = 0

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        running_correct += (logits.argmax(1) == y).sum().item()
        running_count += y.size(0)

    tr_loss = running_loss / running_count
    tr_acc = running_correct / running_count

    te_loss, te_acc = evaluate(model, test_loader, criterion)

    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    test_losses.append(te_loss)
    test_accs.append(te_acc)

    dt = time.time() - t0
    print(f"Epoch {epoch:02d}/{epochs} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"test loss {te_loss:.4f} acc {te_acc:.4f} | "
          f"time {dt:.1f}s")


# -----------------------------
# 7) Save model
# -----------------------------
os.makedirs("checkpoints", exist_ok=True)
ckpt_path = "checkpoints/cifar10_simplecnn.pth"
torch.save(model.state_dict(), ckpt_path)
print("Saved:", ckpt_path)


# -----------------------------
# 8) Plot curves
# -----------------------------
plt.figure()
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)

plt.figure()
plt.plot(train_accs, label="train acc")
plt.plot(test_accs, label="test acc")
plt.legend()
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.tight_layout()
plt.savefig("acc_curve.png", dpi=150)

print("Saved plots: loss_curve.png, acc_curve.png")
