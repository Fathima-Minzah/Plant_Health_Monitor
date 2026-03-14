import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "dataset/PlantVillage"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# 🔥 Evaluate on 2000 random samples only
subset_indices = random.sample(range(len(full_dataset)), 2000)
dataset = Subset(full_dataset, subset_indices)

loader = DataLoader(dataset, batch_size=32)

class_names = full_dataset.classes
NUM_CLASSES = len(class_names)


def evaluate(model, path, model_name):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
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

    print(f"\n--- {model_name} Results ---")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# -------- MobileNet --------
mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = torch.nn.Linear(mobilenet.last_channel, NUM_CLASSES)
evaluate(mobilenet, "models/mobilenet_model.pth", "MobileNetV2")

# -------- ResNet50 --------
resnet = models.resnet50(weights=None)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, NUM_CLASSES)
evaluate(resnet, "models/resnet_model.pth", "ResNet50")

