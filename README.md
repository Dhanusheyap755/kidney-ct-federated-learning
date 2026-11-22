# Federated Learning for Kidney CT Classification

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Multi--Hospital-00BFFF?style=flat)](#)
[![Healthcare AI](https://img.shields.io/badge/Domain-Healthcare%20AI-success?style=flat)](#)

## 📌 Project Overview
This repository implements a privacy-preserving **Federated Learning (FL)** architecture designed for **Kidney Computed Tomography (CT) image classification**. In healthcare systems, patient privacy regulations (such as HIPAA and GDPR) prohibit pooling sensitive medical imaging data into a centralized server. This project simulates a decentralized hospital network where medical institutions collaborate to train deep learning diagnostic models without transmitting raw patient data.

---

## 🎯 Problem Statement & Objectives
- **Data Privacy & Compliance:** Medical facilities cannot share sensitive CT scan imagery across clinical boundaries.
- **Heterogeneous Data Distribution:** Real-world patient cohorts vary across hospital demographics, leading to Independent and Identically Distributed (IID) and Non-IID data distributions.
- **Objectives:**
  1. Simulate isolated multi-hospital clinical environments (`split_hospitals.py`).
  2. Train baseline models isolated to single clinical sites to establish benchmark diagnostic performance (`train_baseline.py`, `train_single_hospital_models.py`).
  3. Implement decentralized model aggregation via Federated Averaging (`train_federated_iid.py`, `train_federated_noniid.py`).
  4. Compare classification accuracy, loss convergence, and generalization capability between isolated and federated paradigms (`results/`).

---

## 🛠️ Technologies Used
- **Language:** Python 3
- **Frameworks & Libraries:** PyTorch / torchvision, NumPy, Pandas, Matplotlib, SciPy
- **Data Pipeline:** Custom dataset partitioning and simulation algorithms
- **Metrics Tracking:** Automated CSV logging for training rounds, test loss, and classification metrics

---

## 🚀 Key Features
- **Decentralized Hospital Simulation:** Partitions CT scan datasets into distinct synthetic clinical client nodes.
- **IID vs. Non-IID Distribution Testing:** Evaluates convergence robustness under both uniform and skewed class balance across hospitals.
- **Isolated Baseline Comparison:** Proves how federated collaborative training surpasses isolated single-institution models in out-of-distribution generalization.
- **Empirical Results Tracking:** Outputs quantitative round-by-round test metrics preserved in `results/federated_iid_results.csv`.

---

## 📂 Project Folder Structure
```
kidney-ct-federated-learning/
├── .gitignore
├── README.md
├── results/
│   └── federated_iid_results.csv      # Empirical round evaluation logs
└── src/
    ├── split_hospitals.py             # Hospital data partitioning simulation
    ├── train_baseline.py              # Centralized benchmark training script
    ├── train_single_hospital_models.py# Isolated single-institution models
    ├── train_federated_iid.py         # Federated training under IID distribution
    └── train_federated_noniid.py      # Federated training under Non-IID skew
```

---

## 💻 Installation & Execution Instructions

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/Dhanusheyap755/kidney-ct-federated-learning.git
cd kidney-ct-federated-learning
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt  # torch, torchvision, numpy, pandas, matplotlib
```

### 2. Partition Hospital Nodes
```bash
python src/split_hospitals.py
```

### 3. Run Federated Training Simulation
```bash
# Federated training with uniform (IID) distribution
python src/train_federated_iid.py

# Federated training with non-uniform (Non-IID) hospital distribution
python src/train_federated_noniid.py
```

---

## 📈 Results & Output
The model generates iterative performance evaluation logs saved directly to `results/federated_iid_results.csv`, tracking round-by-round loss and accuracy across hospital validation sets.

---

## 🔮 Future Enhancements
- Integration of Differential Privacy (DP-SGD) to prevent model inversion attacks.
- Support for 3D volumetric CT scan processing using specialized medical vision backbones (e.g., 3D ResNet, MedNeXt).
- Secure aggregation protocol implementations with homomorphic encryption.
