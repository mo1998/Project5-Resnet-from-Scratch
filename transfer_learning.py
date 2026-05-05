import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import json

# 1. Hardware Detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Transfer Learning running on: {device.upper()}")

# 2. Hyperparameters
batch_size = 64 
lr = 0.001 
num_epochs = 5

# 3. Data Preparation
transform_transfer = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_transfer)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_transfer)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# 4. Load Pretrained Model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)
model = model.to(device)

# 5. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

history = {
    'train_acc': [],
    'test_acc': [],
    'train_loss': []
}

# 6. Training Loop
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}")
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(Loss=f"{running_loss/(i+1):.3f}", Acc=f"{100.*correct/total:.2f}%")
    
    history['train_acc'].append(100. * correct / total)
    history['train_loss'].append(running_loss / len(trainloader))

def test(epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100.*correct/total
    history['test_acc'].append(acc)
    print(f"Test Acc: {acc:.2f}%")

# Run for 5 epochs
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test(epoch)

with open('results/resnet18_transfer.json', 'w') as f:
    json.dump(history, f)
