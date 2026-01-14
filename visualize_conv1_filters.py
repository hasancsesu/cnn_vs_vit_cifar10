import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# --- Same model definition as training ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # conv1 (we will visualize this)
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def save_conv1_grid(model, out_path, title):
    # conv1 weights shape: [32, 3, 3, 3] = [out_channels, in_channels(RGB), kH, kW]
    w = model.features[0].weight.detach().cpu()

    n_filters = w.shape[0]
    cols = 8
    rows = math.ceil(n_filters / cols)

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n_filters):
        filt = w[i]  # [3,3,3]

        # Normalize per-filter to 0..1 for visualization
        fmin = filt.min()
        fmax = filt.max()
        if (fmax - fmin) > 1e-8:
            img = (filt - fmin) / (fmax - fmin)
        else:
            img = torch.zeros_like(filt)

        img = img.permute(1, 2, 0)  # [3,3,3] -> [3,3,3] but HWC for imshow

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img, interpolation="nearest")  # enlarge tiny 3x3 pixels
        ax.set_title(str(i), fontsize=10)
        ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print("Saved:", out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) BEFORE training: fresh random initialization
    model_before = SimpleCNN().to(device)
    save_conv1_grid(model_before, "filters_before_training.png", "Conv1 filters BEFORE training (random)")

    # 2) AFTER training: load your saved weights
    ckpt_path = "checkpoints/cifar10_simplecnn.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_after = SimpleCNN().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_after.load_state_dict(state)
    save_conv1_grid(model_after, "filters_after_training.png", "Conv1 filters AFTER training (learned)")


if __name__ == "__main__":
    main()
