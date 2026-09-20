# CIFAR-10 Image Classification with Regularized Deep CNN (PyTorch)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-84%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

A modular, production-ready Deep Learning pipeline implemented in **PyTorch** to perform multi-class object recognition on the **CIFAR-10** benchmark dataset. The architecture leverages modern regularization strategies, dynamic learning rate scheduling, and Kaiming initialization to prevent overfitting and achieve an **84% Test Accuracy**.


---

## 📌 Project Architecture & Modularization

The codebase follows a clean, modular python structure:
```text
├── model.py            # Deep Neural Network Class (RegularizedCNN Architecture)
├── train.py            # Model Training Pipeline, Data Augmentation & Checkpointing
├── evaluate.py         # Performance Evaluation, Diagnostics & Heatmap Generation
├── best_model.pth      # Optimized Model Checkpoint Weights
├── confusion_matrix.png # Final High-Resolution Evaluation Plot
└── requirements.txt    # Project Dependencies
