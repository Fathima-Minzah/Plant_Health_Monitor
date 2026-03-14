import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "dataset/PlantVillage"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32)

class_names = dataset.classes
NUM_CLASSES = len(class_names)

def evaluate(model, path):
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))

# Evaluate MobileNet
mobilenet = models.mobilenet_v2()
mobilenet.classifier[1] = torch.nn.Linear(mobilenet.last_channel, NUM_CLASSES)
evaluate(mobilenet, "models/mobilenet_model.pth")

# Evaluate ViT
vit = models.vit_b_16()
vit.heads.head = torch.nn.Linear(vit.heads.head.in_features, NUM_CLASSES)
evaluate(vit, "models/vit_model.pth")
