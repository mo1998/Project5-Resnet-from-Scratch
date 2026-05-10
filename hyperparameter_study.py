import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ResNet18
import json
import matplotlib.pyplot as plt
import os
import time

def train_and_evaluate(lr, batch_size, epochs=5):
    """
    Train ResNet18 with specific hyperparameters and return history.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data preparation (Standard CIFAR-100 normalization)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # root='./data' matches existing project structure
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    history = {'train_loss': [], 'test_acc': []}
    
    print(f"\n>>> Experiment: LR={lr}, BatchSize={batch_size}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        avg_loss = running_loss / len(trainloader)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(acc)
        print(f"    Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
        
    return history

def main():
    # 3 distinct experiments + 1 extra to show batch size sensitivity
    experiments = [
        {'lr': 0.1,   'batch_size': 128},
        {'lr': 0.01,  'batch_size': 128},
        {'lr': 0.001, 'batch_size': 128},
        {'lr': 0.1,   'batch_size': 64}
    ]
    
    all_results = {}
    
    for exp in experiments:
        lr = exp['lr']
        bs = exp['batch_size']
        label = f"LR={lr}, BS={bs}"
        history = train_and_evaluate(lr, bs, epochs=5) # 5 epochs is enough to see sensitivity trends
        all_results[label] = history

    # Save numerical results
    os.makedirs('results', exist_ok=True)
    with open('results/hyperparameter_sensitivity.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Generate Plots
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    for label, history in all_results.items():
        plt.plot(range(1, 6), history['test_acc'], marker='o', label=label)
    plt.title('Sensitivity: Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    for label, history in all_results.items():
        plt.plot(range(1, 6), history['train_loss'], marker='x', label=label)
    plt.title('Sensitivity: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_sensitivity.png')
    print("\n[SUCCESS] Sensitivity study complete.")
    print("Numerical results: results/hyperparameter_sensitivity.json")
    print("Visual comparison: hyperparameter_sensitivity.png")

if __name__ == "__main__":
    main()
