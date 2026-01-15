import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# This connects to the file you just created!
from vit_model import VisionTransformer 

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. DATA AUGMENTATION (From our earlier discussion)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # 2. INITIALIZE MODEL
    model = VisionTransformer(img_size=64, n_classes=10).to(device)

    # 3. OPTIMIZER & SCHEDULER (The "Fine-tunings")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 4. TRAINING LOOP
    print("Starting Training...")
    for epoch in range(10):  # You can change the number of epochs
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1} - Loss: {running_loss/len(trainloader):.4f}")

    # 5. SAVE THE WEIGHTS
    torch.save(model.state_dict(), "vit_cifar10_weights.pth")
    print("Finished Training and Saved Model.")

if __name__ == "__main__":
    train()