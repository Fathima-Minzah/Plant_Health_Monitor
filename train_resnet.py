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
EPOCHS = 2

print("Using device:", DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Use subset (5000 images for speed)
subset_indices = random.sample(range(len(full_dataset)), 5000)
dataset = Subset(full_dataset, subset_indices)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

NUM_CLASSES = len(full_dataset.classes)

# -------- ResNet50 --------
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet = resnet.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    resnet.train()
    running_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            print(f"Batch {i}, Loss: {loss.item()}")

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

torch.save(resnet.state_dict(), "models/resnet_model.pth")
print("ResNet50 saved successfully!")
