import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from model import ResNet18
from tqdm import tqdm
import json

# 1. Hardware & Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
lr = 0.1
num_epochs = 20

# ... (Cutout and Mixup classes/functions same as before)
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 4. Data Preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    Cutout(n_holes=1, length=16),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# 5. Model, Criterion, Optimizer
net = ResNet18(num_classes=100, use_bn=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

history = {
    'train_loss': [],
    'test_acc': []
}

# 6. Training Loop
def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, use_cuda=(device=='cuda'))
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(Loss=f"{train_loss/(batch_idx+1):.3f}")
    
    history['train_loss'].append(train_loss / len(trainloader))

def test(epoch):
    global best_acc
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

    acc = 100.*correct/total
    history['test_acc'].append(acc)
    print(f'Test Acc: {acc:.2f}%')
    return acc

best_acc = 0
for epoch in range(1, num_epochs + 1):
    train(epoch)
    acc = test(epoch)
    scheduler.step()
    if acc > best_acc:
        print('Saving..')
        state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch}
        torch.save(state, 'best_advanced_resnet18.pth')
        best_acc = acc

with open('results/resnet18_advanced.json', 'w') as f:
    json.dump(history, f)

print(f"Finished. Best accuracy: {best_acc:.2f}%")
