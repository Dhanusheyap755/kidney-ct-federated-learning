"""
Phase 4 (Step 3): Federated Learning - Non-IID Simulation

FedKidneyNet
------------
Implements real Federated Learning using FedAvg over
Hospital_A, Hospital_B and Hospital_C (Non-IID).

Outputs:
- federated_noniid_model.pth
- federated_noniid_results.csv
- predictions.csv
- best_model.pth
- accuracy_rounds.png
- loss_rounds.png
- confusion_matrix.png

Author: FedKidneyNet
"""

import os
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

##############################################################
# DEVICE
##############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("FedKidneyNet : Non-IID Federated Learning")
print("Using Device :", device)
print("="*60)

##############################################################
# CONFIGURATION
##############################################################

HOSPITALS_DIR = "../data/hospitals"

HOSPITALS = [
    "Hospital_A",
    "Hospital_B",
    "Hospital_C"
]

NUM_CLIENTS = len(HOSPITALS)

NUM_ROUNDS = 2
LOCAL_EPOCHS = 3

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

RESULT_DIR = "../results"

os.makedirs(RESULT_DIR, exist_ok=True)

##############################################################
# IMAGE PREPROCESSING
##############################################################

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

##############################################################
# LOAD EACH HOSPITAL
##############################################################

client_datasets = []
client_loaders = []

print("\nLoading Hospital Datasets...\n")

for hospital in HOSPITALS:

    folder = os.path.join(HOSPITALS_DIR,hospital)

    dataset = datasets.ImageFolder(
        root=folder,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    client_datasets.append(dataset)
    client_loaders.append(loader)

    print(
        hospital,
        "->",
        len(dataset),
        "images"
    )

class_names = client_datasets[0].classes
num_classes = len(class_names)

print("\nClasses:",class_names)

##############################################################
# COMMON TEST SET
##############################################################

print("\nPreparing Common Test Set...")

test_sets = []

for dataset in client_datasets:

    test_size = int(0.15*len(dataset))

    train_size = len(dataset)-test_size

    _,test = random_split(
        dataset,
        [train_size,test_size]
    )

    test_sets.append(test)

common_test = ConcatDataset(test_sets)

test_loader = DataLoader(
    common_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test Images :",len(common_test))

##############################################################
# MODEL
##############################################################

def build_model():

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model.to(device)

##############################################################
# LOCAL TRAINING
##############################################################

def train_local(model,loader):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    running_loss = 0

    batches = 0

    for epoch in range(LOCAL_EPOCHS):

        for images,labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs,labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            batches += 1

    average_loss = running_loss/max(batches,1)

    return model.state_dict(),average_loss

##############################################################
# FEDERATED AVERAGING
##############################################################

def fedavg(state_dicts,client_sizes):

    total_samples = sum(client_sizes)

    avg = copy.deepcopy(state_dicts[0])

    for key in avg.keys():

        if avg[key].dtype.is_floating_point:

            avg[key] *= client_sizes[0]/total_samples

            for i in range(1,len(state_dicts)):

                avg[key] += (
                    state_dicts[i][key]
                    * client_sizes[i]/total_samples
                )

    return avg
    ##############################################################
# EVALUATION
##############################################################

def evaluate(model, loader):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_preds
    )


##############################################################
# SAVE PREDICTIONS
##############################################################

def save_predictions(labels, predictions):

    prediction_file = os.path.join(
        RESULT_DIR,
        "predictions.csv"
    )

    with open(prediction_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Actual",
            "Predicted"
        ])

        for gt, pred in zip(labels, predictions):
            writer.writerow([gt, pred])

    print("Saved:", prediction_file)


##############################################################
# SAVE CONFUSION MATRIX
##############################################################

def save_confusion_matrix(labels, predictions):

    cm = confusion_matrix(labels, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(6,6))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    figure = os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        figure,
        dpi=300
    )

    plt.close()

    print("Saved:", figure)


##############################################################
# FEDERATED TRAINING
##############################################################

global_model = build_model()

global_weights = global_model.state_dict()

client_sizes = [
    len(ds)
    for ds in client_datasets
]

results = []

best_accuracy = 0.0

print("\nStarting Federated Learning...\n")

for round_number in range(1, NUM_ROUNDS + 1):

    local_weights = []

    local_losses = []

    print("="*60)
    print(
        f"Communication Round {round_number}/{NUM_ROUNDS}"
    )
    print("="*60)

    ##########################################################
    # LOCAL TRAINING
    ##########################################################

    for idx, loader in enumerate(client_loaders):

        print(
            f"Training Client {idx+1}"
        )

        client_model = build_model()

        client_model.load_state_dict(
            global_weights
        )

        weights, loss = train_local(
            client_model,
            loader
        )

        local_weights.append(weights)

        local_losses.append(loss)

    ##########################################################
    # FEDAVG
    ##########################################################

    global_weights = fedavg(
        local_weights,
        client_sizes
    )

    global_model.load_state_dict(
        global_weights
    )

    ##########################################################
    # EVALUATE GLOBAL MODEL
    ##########################################################

    accuracy, precision, recall, f1, labels, preds = evaluate(
        global_model,
        test_loader
    )

    average_loss = np.mean(local_losses)

    print(
        f"Loss      : {average_loss:.4f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    results.append({

        "round": round_number,

        "train_loss": average_loss,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1

    })

    ##########################################################
    # SAVE BEST MODEL
    ##########################################################

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(

            global_model.state_dict(),

            os.path.join(

                RESULT_DIR,

                "best_federated_noniid_model.pth"

            )

        )

        print("Best model updated.")
        ##############################################################
# SAVE RESULTS CSV
##############################################################

results_file = os.path.join(
    RESULT_DIR,
    "federated_noniid_results.csv"
)

with open(results_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "round",
            "train_loss",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    writer.writeheader()

    writer.writerows(results)

print("\nSaved:", results_file)

##############################################################
# SAVE FINAL MODEL
##############################################################

torch.save(

    global_model.state_dict(),

    os.path.join(
        RESULT_DIR,
        "federated_noniid_model.pth"
    )

)

print("Saved: federated_noniid_model.pth")

##############################################################
# SAVE PREDICTIONS
##############################################################

save_predictions(labels, preds)

##############################################################
# SAVE CONFUSION MATRIX
##############################################################

save_confusion_matrix(labels, preds)

##############################################################
# ACCURACY GRAPH
##############################################################

rounds = [r["round"] for r in results]

accuracies = [
    r["accuracy"] * 100
    for r in results
]

plt.figure(figsize=(8,5))

plt.plot(
    rounds,
    accuracies,
    marker="o",
    linewidth=2
)

plt.xlabel("Communication Round")
plt.ylabel("Accuracy (%)")
plt.title("Federated Learning Accuracy")

plt.grid(True)

plt.tight_layout()

accuracy_graph = os.path.join(
    RESULT_DIR,
    "accuracy_rounds.png"
)

plt.savefig(
    accuracy_graph,
    dpi=300
)

plt.close()

print("Saved:", accuracy_graph)

##############################################################
# LOSS GRAPH
##############################################################

losses = [
    r["train_loss"]
    for r in results
]

plt.figure(figsize=(8,5))

plt.plot(
    rounds,
    losses,
    marker="o",
    linewidth=2
)

plt.xlabel("Communication Round")
plt.ylabel("Training Loss")
plt.title("Training Loss Across Communication Rounds")

plt.grid(True)

plt.tight_layout()

loss_graph = os.path.join(
    RESULT_DIR,
    "loss_rounds.png"
)

plt.savefig(
    loss_graph,
    dpi=300
)

plt.close()

print("Saved:", loss_graph)

##############################################################
# COMMUNICATION ROUND CSV
##############################################################

communication_file = os.path.join(
    RESULT_DIR,
    "communication_rounds.csv"
)

with open(
    communication_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Round",
        "Accuracy",
        "Loss"
    ])

    for r in results:

        writer.writerow([
            r["round"],
            r["accuracy"],
            r["train_loss"]
        ])

print("Saved:", communication_file)

##############################################################
# FINAL SUMMARY
##############################################################

print("\n" + "="*70)
print("FedKidneyNet Non-IID Training Completed Successfully")
print("="*70)

print(f"Best Accuracy : {best_accuracy*100:.2f}%")
print(f"Communication Rounds : {NUM_ROUNDS}")
print(f"Clients : {NUM_CLIENTS}")

print("\nGenerated Files")

print("-------------------------------------------")

print("✓ federated_noniid_model.pth")

print("✓ best_federated_noniid_model.pth")

print("✓ federated_noniid_results.csv")

print("✓ communication_rounds.csv")

print("✓ predictions.csv")

print("✓ confusion_matrix.png")

print("✓ accuracy_rounds.png")

print("✓ loss_rounds.png")

print("\nResults Folder :", RESULT_DIR)

print("="*70)