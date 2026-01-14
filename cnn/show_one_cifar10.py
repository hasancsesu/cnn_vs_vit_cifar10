import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

ds = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False,
    transform=T.ToTensor()
)

img, label = ds[0]          # pick one image
img = img.permute(1, 2, 0)  # CHW → HWC for display

plt.imshow(img)
plt.title(f"Label: {ds.classes[label]}")
plt.axis("off")
plt.show()

