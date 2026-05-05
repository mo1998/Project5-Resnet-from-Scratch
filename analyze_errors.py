import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import ResNet18 as CustomResNet18
from torchvision.models import resnet18 as TorchResNet18
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# 1. CIFAR-100 Superclass Mapping
fine_to_superclass = {
    0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
    10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
    20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
    30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
    40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
    50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
    60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
    70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
    80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
    90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13
}

superclass_names = [
    'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
    'household electrical devices', 'household furniture', 'insects', 'large carnivores',
    'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
    'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
    'vehicles 1', 'vehicles 2'
]

parser = argparse.ArgumentParser(description='CIFAR-100 Error Analysis')
parser.add_argument('--path', type=str, default='best_resnet18_cifar100.pth', help='Path to model checkpoint')
args = parser.parse_args()

# 2. Hardware & Data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Detect if we should use 224 or 32 based on the model (Torchvision ResNet usually trained on 224)
# We'll default to 224 if it's the transfer model, 32 if custom
checkpoint_path = args.path
if not os.path.exists(checkpoint_path):
    print(f"Error: {checkpoint_path} not found.")
    exit()

state_dict = torch.load(checkpoint_path, map_location=device)
if 'net' in state_dict:
    sd = state_dict['net']
else:
    sd = state_dict

# Heuristic to detect architecture
is_torchvision = 'maxpool.weight' in sd or 'fc.weight' in sd and 'linear.weight' not in sd

if is_torchvision:
    print("Detected Torchvision architecture")
    model = TorchResNet18()
    model.fc = nn.Linear(model.fc.in_features, 100)
    transform_size = 224
else:
    print("Detected Custom architecture")
    model = CustomResNet18(num_classes=100, use_bn=True)
    transform_size = 32

model.load_state_dict(sd)
model = model.to(device)
model.eval()

transform_test = transforms.Compose([
    transforms.Resize(transform_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 4. Collect Predictions
all_preds = []
all_targets = []

print("Running evaluation...")
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# 5. Fine-grained Analysis
cm_fine = confusion_matrix(all_targets, all_preds)
accuracy_fine = np.trace(cm_fine) / np.sum(cm_fine)
print(f"Fine-grained Accuracy: {accuracy_fine*100:.2f}%")

# 6. Superclass Analysis
superclass_preds = np.array([fine_to_superclass[p] for p in all_preds])
superclass_targets = np.array([fine_to_superclass[t] for t in all_targets])

cm_super = confusion_matrix(superclass_targets, superclass_preds)
accuracy_super = np.trace(cm_super) / np.sum(cm_super)
print(f"Superclass Accuracy: {accuracy_super*100:.2f}%")

# 7. Visualization
plt.figure(figsize=(15, 12))
sns.heatmap(cm_super, annot=True, fmt='d', xticklabels=superclass_names, yticklabels=superclass_names)
plt.title(f'Superclass Confusion Matrix ({os.path.basename(checkpoint_path)})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('superclass_confusion.png')
print("Saved superclass confusion matrix as 'superclass_confusion.png'")

# Find top confused fine classes
misclassified = all_preds != all_targets
confusions = list(zip(all_targets[misclassified], all_preds[misclassified]))
from collections import Counter
most_confused = Counter(confusions).most_common(10)

print("\nTop 10 most frequent fine-class confusions:")
classes = testset.classes
for (actual, pred), count in most_confused:
    print(f"Actual: {classes[actual]:<15} | Predicted: {classes[pred]:<15} | Count: {count}")
