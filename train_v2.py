import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 1. Hardware & Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64  # Adjusted for 8GB RAM/VRAM
num_epochs = 10
learning_rate = 0.001

# 2. Advanced Data Augmentation (Week 3 Requirement)
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Data Loaders
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# 4. Model Setup (Transfer Learning)
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# 5. Optimization & Scheduling (Week 3 Requirement)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Cosine Annealing reduces LR gradually for a better "settle" in accuracy
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 6. Training & Validation Loop
def run_epoch(epoch, loader, is_train=True):
    if is_train:
        model.train()
        desc = f"Train Epoch {epoch}"
    else:
        model.eval()
        desc = f"Test  Epoch {epoch}"
        
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc)
    
    with torch.set_grad_enabled(is_train):
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if is_train:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(Loss=f"{running_loss/(i+1):.3f}", Acc=f"{100.*correct/total:.2f}%")
            
    return 100. * correct / total

# 7. Main Execution
best_acc = 0
for epoch in range(1, num_epochs + 1):
    train_acc = run_epoch(epoch, trainloader, is_train=True)
    test_acc = run_epoch(epoch, testloader, is_train=False)
    
    # Update Scheduler
    scheduler.step()
    
    # Save best model
    if test_acc > best_acc:
        print(f"✅ New Best Test Acc: {test_acc:.2f}% | Saving model...")
        torch.save(model.state_dict(), 'best_resnet18_cifar100.pth')
        best_acc = test_acc

print(f"\nTraining Complete. Best Accuracy achieved: {best_acc:.2f}%")