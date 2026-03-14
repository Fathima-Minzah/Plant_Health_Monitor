import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import random

os.makedirs("models", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "dataset/PlantVillage"
BATCH_SIZE = 16
EPOCHS = 2   # keep small for CPU

print("Using device:", DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# 🔥 Use only 5000 random samples
subset_indices = random.sample(range(len(full_dataset)), 5000)
dataset = Subset(full_dataset, subset_indices)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

NUM_CLASSES = len(full_dataset.classes)

def train_model(model, name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")

        print(f"{name} Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), f"models/{name}.pth")
    print(f"{name} saved successfully!")


# -------- MobileNet --------
mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, NUM_CLASSES)

train_model(mobilenet, "mobilenet_model")


