import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import ResNet18
from tqdm import tqdm
import os
import argparse
import json

# 1. Hardware & Hyperparameters
parser = argparse.ArgumentParser(description='ResNet-18 CIFAR-100 BN Comparison')
parser.add_argument('--aug', action='store_true', help='Use data augmentation (RandomCrop, RandomHorizontalFlip)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per experiment')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
lr = 0.1
num_epochs = args.epochs

print(f"Using device: {device} | Augmentation: {args.aug}")

# 2. Data Preparation
if args.aug:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

def train_model(use_bn):
    net = ResNet18(num_classes=100, use_bn=use_bn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    history = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': []
    }
    
    for epoch in range(1, num_epochs + 1):
        # Train
        net.train()
        correct = 0
        total = 0
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"BN={use_bn} Epoch {epoch}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix(Loss=f"{running_loss/total*inputs.size(0):.3f}")
        
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(trainloader)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(avg_loss)
        
        # Test
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        history['test_acc'].append(test_acc)
        print(f"BN={use_bn} | Epoch {epoch} | Loss: {avg_loss:.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    return history

# Run experiments
suffix = "aug" if args.aug else "noaug"
os.makedirs('results', exist_ok=True)

print(f"Starting experiment: WITHOUT Batch Normalization (Augmentation: {args.aug})...")
history_no_bn = train_model(use_bn=False)
with open(f'results/resnet18_no_bn_{suffix}.json', 'w') as f:
    json.dump(history_no_bn, f)

print(f"\nStarting experiment: WITH Batch Normalization (Augmentation: {args.aug})...")
history_bn = train_model(use_bn=True)
with open(f'results/resnet18_with_bn_{suffix}.json', 'w') as f:
    json.dump(history_bn, f)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Accuracy
ax1.plot(history_no_bn['train_acc'], label='Train (No BN)', linestyle=':', alpha=0.6, color='blue')
ax1.plot(history_no_bn['test_acc'], label='Test (No BN)', linestyle=':', linewidth=2, color='blue')
ax1.plot(history_bn['train_acc'], label='Train (BN)', linestyle='-', alpha=0.6, color='orange')
ax1.plot(history_bn['test_acc'], label='Test (BN)', linestyle='-', linewidth=2, color='orange')
ax1.set_title(f'Accuracy Comparison\n(Augmentation: {args.aug})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Subplot 2: Convergence Speed (Loss)
ax2.plot(history_no_bn['train_loss'], label='Loss (No BN)', linestyle=':', linewidth=2, color='blue')
ax2.plot(history_bn['train_loss'], label='Loss (BN)', linestyle='-', linewidth=2, color='orange')
ax2.set_title('Convergence Speed (Training Loss)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross Entropy Loss')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f'bn_comparison_{suffix}.png')
print(f"\nExperiment complete. Plot saved as 'bn_comparison_{suffix}.png'.")
