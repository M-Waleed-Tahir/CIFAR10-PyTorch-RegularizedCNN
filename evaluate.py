# ==============================================================================
# DigiHust AI Internship Program - Assignment A03
# Student: Muhammad Waleed Tahir | ID: DGH2600170
# File: evaluate.py (Metric Diagnostics & Visual Plots)
# ==============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import RegularizedCNN

def evaluate_performance():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    model = RegularizedCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(targets.numpy())
            y_pred.extend(predicted.cpu().numpy())

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("================ Model Performance Report ================")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('CIFAR-10 Test Confusion Matrix — DGH2600170')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    evaluate_performance()
