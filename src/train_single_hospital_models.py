"""
Phase 3 (Step 3): Train 3 separate single-hospital models
Then test each one on ALL 3 hospitals' test data to reveal the
generalization gap (a model trained only on Hospital_A's data
often performs worse on Hospital_B/C's data).

This produces the "single-hospital" baseline numbers you'll later
compare against your federated (FedAvg) model.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

HOSPITALS_DIR = "../data/hospitals"
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C"]
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.0001

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------
# 1. Load each hospital's dataset, split into train/test
# ---------------------------
hospital_datasets = {}
hospital_train_loaders = {}
hospital_test_loaders = {}

for hospital in HOSPITALS:
    path = os.path.join(HOSPITALS_DIR, hospital)
    full_dataset = datasets.ImageFolder(root=path, transform=transform)
    hospital_datasets[hospital] = full_dataset

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    hospital_train_loaders[hospital] = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    hospital_test_loaders[hospital] = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"{hospital}: {len(full_dataset)} total images "
          f"(classes: {full_dataset.classes})")

num_classes = len(hospital_datasets[HOSPITALS[0]].classes)

# ---------------------------
# 2. Training / evaluation helper functions
# ---------------------------
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

def train_model(model, train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Train Acc: {correct/total:.4f}")
    return model

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ---------------------------
# 3. Train one model per hospital, save it
# ---------------------------
trained_models = {}

for hospital in HOSPITALS:
    print(f"\n=== Training model on {hospital} only ===")
    model = build_model()
    model = train_model(model, hospital_train_loaders[hospital], EPOCHS)
    trained_models[hospital] = model
    torch.save(model.state_dict(), f"../{hospital}_model.pth")
    print(f"Saved {hospital}_model.pth")

# ---------------------------
# 4. Cross-hospital evaluation matrix
# ---------------------------
print("\n\n=== CROSS-HOSPITAL GENERALIZATION RESULTS ===")
print("(rows = model trained on, columns = tested on)\n")

results = {}
header = "Trained on \\ Tested on".ljust(22) + "".join(h.ljust(14) for h in HOSPITALS)
print(header)

for train_hospital in HOSPITALS:
    row = f"{train_hospital}".ljust(22)
    results[train_hospital] = {}
    for test_hospital in HOSPITALS:
        acc = evaluate(trained_models[train_hospital], hospital_test_loaders[test_hospital])
        results[train_hospital][test_hospital] = acc
        row += f"{acc:.4f}".ljust(14)
    print(row)

print("\nDiagonal values (e.g. Hospital_A trained + tested on Hospital_A) = 'home' accuracy")
print("Off-diagonal values (e.g. Hospital_A trained, tested on Hospital_B) = generalization gap")
print("\nDone. These numbers are your 'single-hospital' comparison baseline.")
