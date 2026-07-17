"""
Phase 4 (Step 3): Federated Learning - Non-IID Simulation
Reuses your existing skewed Hospital_A/B/C folders (data/hospitals/)
and actually runs FedAvg across them (real federation, not separate models).

Logs Accuracy, Precision, Recall, F1, and average training loss per round
-> everything needed for Table III, Table IV, Fig 8, Fig 9, Fig 10.
"""

import os
import copy
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 1. Config
# ---------------------------
HOSPITALS_DIR = "../data/hospitals"
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C"]
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------
# 2. Load each hospital's data as a separate client (non-IID, skewed)
# ---------------------------
client_train_loaders = []
client_datasets = []

for hospital in HOSPITALS:
    path = os.path.join(HOSPITALS_DIR, hospital)
    ds = datasets.ImageFolder(root=path, transform=transform)
    client_datasets.append(ds)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    client_train_loaders.append(loader)
    print(f"{hospital}: {len(ds)} images (non-IID, classes: {ds.classes})")

class_names = client_datasets[0].classes
num_classes = len(class_names)

# ---------------------------
# 3. Build a COMMON test set (combine a held-out slice from all 3 hospitals)
#    so the global model is judged fairly across the whole class spectrum
# ---------------------------
test_subsets = []
for ds in client_datasets:
    test_size = int(0.15 * len(ds))
    _, test_part = random_split(ds, [len(ds) - test_size, test_size])
    test_subsets.append(test_part)

common_test_set = ConcatDataset(test_subsets)
test_loader = DataLoader(common_test_set, batch_size=BATCH_SIZE, shuffle=False)
print(f"Common test set size: {len(common_test_set)}")

# ---------------------------
# 4. Model builder
# ---------------------------
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

# ---------------------------
# 5. Local training (returns updated weights + average loss)
# ---------------------------
def train_local(model, loader, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_loss, batches = 0.0, 0
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
    avg_loss = total_loss / max(batches, 1)
    return model.state_dict(), avg_loss

# ---------------------------
# 6. FedAvg aggregation (weighted by client dataset size - fairer for non-IID)
# ---------------------------
def federated_average(state_dicts, weights):
    total = sum(weights)
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        if avg_state[key].dtype.is_floating_point:
            avg_state[key] = avg_state[key] * (weights[0] / total)
            for i in range(1, len(state_dicts)):
                avg_state[key] += state_dicts[i][key] * (weights[i] / total)
    return avg_state

# ---------------------------
# 7. Evaluation
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
# 8. Federated training loop (Non-IID)
# ---------------------------
global_model = build_model()
global_weights = global_model.state_dict()
client_sizes = [len(ds) for ds in client_datasets]

results_log = []

print("\n=== Starting Federated Training (Non-IID) ===")
for rnd in range(1, NUM_ROUNDS + 1):
    local_state_dicts = []
    round_losses = []

    for loader in client_train_loaders:
        local_model = build_model()
        local_model.load_state_dict(global_weights)
        updated_state, avg_loss = train_local(local_model, loader, LOCAL_EPOCHS)
        local_state_dicts.append(updated_state)
        round_losses.append(avg_loss)

    global_weights = federated_average(local_state_dicts, client_sizes)
    global_model.load_state_dict(global_weights)

    acc, prec, rec, f1 = evaluate(global_model, test_loader)
    mean_loss = sum(round_losses) / len(round_losses)

    print(f"Round {rnd}/{NUM_ROUNDS} | Avg Train Loss: {mean_loss:.4f} | "
          f"Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    results_log.append({"round": rnd, "train_loss": mean_loss, "accuracy": acc,
                          "precision": prec, "recall": rec, "f1": f1})

# ---------------------------
# 9. Save results + model
# ---------------------------
os.makedirs("../results", exist_ok=True)
with open("../results/federated_noniid_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["round", "train_loss", "accuracy", "precision", "recall", "f1"])
    writer.writeheader()
    writer.writerows(results_log)

torch.save(global_model.state_dict(), "../federated_noniid_model.pth")
print("\nSaved: ../federated_noniid_model.pth")
print("Saved: ../results/federated_noniid_results.csv  (Table III & IV / Fig 8, 9, 10)")