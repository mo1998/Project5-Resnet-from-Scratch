import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import ResNet18 as CustomResNet18
from torchvision.models import resnet18 as TorchResNet18
import argparse
import os

def verify_architecture(model_type='custom', use_bn=True):
    print("\n--- Architecture Verification ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Initialize Model
    if model_type == 'custom':
        net = CustomResNet18(num_classes=100, use_bn=use_bn).to(device)
        print(f"Verifying Custom ResNet18 (use_bn={use_bn})...")
    else:
        net = TorchResNet18()
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 100)
        net = net.to(device)
        print("Verifying Torchvision ResNet18...")

    # 2. Test Forward Pass Shape
    input_size = (1, 3, 32, 32) if model_type == 'custom' else (1, 3, 224, 224)
    dummy_input = torch.randn(input_size).to(device)
    try:
        output = net(dummy_input)
        print(f"✓ Forward pass successful. Input {input_size} -> Output {output.shape}")
        assert output.shape == (1, 100), f"Expected output (1, 100), got {output.shape}"
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return

    # 3. Parameter Count
    total_params = sum(p.numel() for p in net.parameters())
    print(f"✓ Total parameters: {total_params:,}")

    # 4. CIFAR-100 Specific Optimization Check (for custom)
    if model_type == 'custom':
        # Check first conv layer
        conv1_kernel = net.conv1.kernel_size
        print(f"✓ Initial Conv kernel: {conv1_kernel[0]}x{conv1_kernel[1]} (Expected 3x3 for CIFAR)")
        assert conv1_kernel == (3, 3), "CIFAR optimization expects 3x3 conv1"

        # Check BatchNorm presence
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in net.modules())
        if use_bn:
            print(f"✓ BatchNorm layers detected: {has_bn}")
            assert has_bn, "Expected BatchNorm layers but none found"
        else:
            print(f"✓ BatchNorm layers absent: {not has_bn}")
            assert not has_bn, "Found BatchNorm layers despite use_bn=False"

    print("--- Verification Complete ---\n")

def test_model(model_path, model_type='custom', use_bn=True):
    # 1. Hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")

    # 2. Data Preparation
    if model_type == 'custom':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        # Standard normalization for pretrained models
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 3. Initialize Model
    if model_type == 'custom':
        net = CustomResNet18(num_classes=100, use_bn=use_bn).to(device)
    else:
        net = TorchResNet18()
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 100)
        net = net.to(device)
    
    # 4. Load Weights
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint is a state_dict or a full model
    if isinstance(checkpoint, dict):
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Strip 'module.' prefix if it was saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        net.load_state_dict(new_state_dict)
    else:
        net = checkpoint
    
    net.eval()
    
    # 5. Testing Loop
    correct = 0
    total = 0
    print(f"Starting test on {len(testset)} images...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 20 == 0:
                print(f'Batch {batch_idx + 1}/{len(testloader)} | Acc: {100.*correct/total:.2f}%')

    print(f'\nFinal Test Accuracy: {100.*correct/total:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ResNet CIFAR-100 Testing & Verification')
    parser.add_argument('--model', type=str, default='../best_advanced_resnet18.pth', help='path to model weights')
    parser.add_argument('--type', type=str, choices=['custom', 'torchvision'], default='custom', help='model architecture type')
    parser.add_argument('--no-bn', action='store_true', help='disable batch normalization (only for custom type)')
    parser.add_argument('--verify', action='store_true', help='run architecture verification checks')
    args = parser.parse_args()

    if args.verify:
        verify_architecture(model_type=args.type, use_bn=not args.no_bn)

    # Only run full test if a model path exists or we didn't just want verification
    if os.path.exists(args.model):
        test_model(args.model, model_type=args.type, use_bn=not args.no_bn)
    elif not args.verify:
        print(f"Error: Model file {args.model} not found and --verify not set.")
