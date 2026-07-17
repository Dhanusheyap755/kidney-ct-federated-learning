"""
Phase 4 (Step 2): Federated Learning - IID Simulation
Manually implements FedAvg (Federated Averaging) - no external FL framework needed.

Splits the full dataset EVENLY (IID) across 3 clients, trains each locally,
averages their weights each round, and evaluates the global model.

Also computes Accuracy, Precision, Recall, F1-score (needed for your paper's Table III).
"""

import os
import copy
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 1. Config
# ---------------------------
DATA_DIR = "../data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
NUM_CLIENTS = 3
NUM_ROUNDS = 10          # communication rounds (server <-> clients)
LOCAL_EPOCHS = 1         # how many epochs each client trains per round
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------
# 2. Load full dataset, hold out a common test set
# ---------------------------
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names} | Total images: {len(full_dataset)}")

test_size = int(0.2 * len(full_dataset))
train_pool_size = len(full_dataset) - test_size
train_pool, test_set = random_split(full_dataset, [train_pool_size, test_size])

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# 3. IID split: divide train_pool evenly & randomly across clients
# ---------------------------
indices = train_pool.indices
np.random.seed(42)
np.random.shuffle(indices)
client_indices = np.array_split(indices, NUM_CLIENTS)

client_loaders = []
for i, idxs in enumerate(client_indices):
    subset = Subset(full_dataset, idxs.tolist())
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    client_loaders.append(loader)
    print(f"Client {i+1}: {len(idxs)} images (IID split)")

# ---------------------------
# 4. Model builder
# ---------------------------
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

# ---------------------------
# 5. Local training (one client, one round)
# ---------------------------
def train_local(model, loader, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# ---------------------------
# 6. FedAvg aggregation
# ---------------------------
def federated_average(state_dicts):
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        if avg_state[key].dtype.is_floating_point:
            for i in range(1, len(state_dicts)):
                avg_state[key] += state_dicts[i][key]
            avg_state[key] = avg_state[key] / len(state_dicts)
    return avg_state

# ---------------------------
# 7. Evaluation (Accuracy, Precision, Recall, F1)
# ---------------------------
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    return acc, precision, recall, f1

# ---------------------------
# 8. Federated training loop
# ---------------------------
global_model = build_model()
global_weights = global_model.state_dict()

results_log = []  # for Table IV (accuracy per round)

print("\n=== Starting Federated Training (IID) ===")
for rnd in range(1, NUM_ROUNDS + 1):
    local_state_dicts = []

    for i, loader in enumerate(client_loaders):
        local_model = build_model()
        local_model.load_state_dict(global_weights)
        updated_state = train_local(local_model, loader, LOCAL_EPOCHS)
        local_state_dicts.append(updated_state)

    # Aggregate
    global_weights = federated_average(local_state_dicts)
    global_model.load_state_dict(global_weights)

    # Evaluate global model after this round
    acc, prec, rec, f1 = evaluate(global_model, test_loader)
    print(f"Round {rnd}/{NUM_ROUNDS} | Acc: {acc:.4f} | Precision: {prec:.4f} | "
          f"Recall: {rec:.4f} | F1: {f1:.4f}")

    results_log.append({"round": rnd, "accuracy": acc, "precision": prec,
                          "recall": rec, "f1": f1})

# ---------------------------
# 9. Save results + model
# ---------------------------
os.makedirs("../results", exist_ok=True)
with open("../results/federated_iid_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "precision", "recall", "f1"])
    writer.writeheader()
    writer.writerows(results_log)

torch.save(global_model.state_dict(), "../federated_iid_model.pth")
print("\nSaved: ../federated_iid_model.pth")
print("Saved: ../results/federated_iid_results.csv  (use this for Table III & IV / Fig 8 & 9)")
