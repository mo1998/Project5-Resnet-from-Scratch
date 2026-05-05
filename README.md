# ResNet from Scratch vs. Transfer Learning on CIFAR-100

This project explores the implementation of ResNet-18, the impact of Batch Normalization, data augmentation, and a comparison with transfer learning.

## Project Structure

- `model.py`: ResNet-18 implementation from scratch.
- `train.py`: Basic training script for the scratch model.
- `train_v2.py`: Refined training script using transfer learning and advanced augmentation. Saves `best_resnet18_cifar100.pth`.
- `compare_bn.py`: Experiment comparing training with and without Batch Normalization (Week 1 & 2). Supports toggling augmentation and setting epochs.
- `transfer_learning.py`: Script to fine-tune a pretrained ResNet-18 on CIFAR-100 (Week 2). Logs both accuracy and training loss.
- `train_advanced.py`: Advanced training with Cutout, Mixup, and Cosine Annealing (Week 3).
- `analyze_errors.py`: Detailed error analysis at fine-class and superclass levels (Week 4).
- `final_comparison.py`: Automatically aggregates all JSON results from `results/` to generate a dual-plot comparison (Accuracy and Convergence Speed).
- `results/`: Directory containing JSON logs of experiment metrics.

## How to Run

### 0. Quick Start (Jupyter Notebook)
For a guided walkthrough of the entire project with visualizations, open:
`ResNet_Project_Walkthrough.ipynb`

### 1. Batch Normalization & Convergence Comparison
Run the comparison between ResNet-18 with and without BN. You can toggle augmentation to see how it interacts with BN:

**Without Augmentation:**
```bash
python compare_bn.py --epochs 10
```
Logs to `results/resnet18_no_bn_noaug.json` and `results/resnet18_with_bn_noaug.json`. Generates `bn_comparison_noaug.png`.

**With Augmentation:**
```bash
python compare_bn.py --aug --epochs 10
```
Logs to `results/resnet18_no_bn_aug.json` and `results/resnet18_with_bn_aug.json`. Generates `bn_comparison_aug.png`.

### 2. Transfer Learning (Standard & Refined)
**Basic Transfer Learning:**
```bash
python transfer_learning.py
```
Logs results to `results/resnet18_transfer.json`.

**Refined Transfer Learning (v2):**
```bash
python train_v2.py
```
Uses advanced augmentation and saves the best model as `best_resnet18_cifar100.pth`.

### 3. Advanced Training (Week 3)
Train with Cutout and Mixup:
```bash
python train_advanced.py
```
This saves the best model as `best_advanced_resnet18.pth` and logs to `results/resnet18_advanced.json`.

### 4. Comprehensive Comparison (The "Week 2 & 4" View)
After running the above experiments, generate a unified comparison plot showing both **Accuracy** and **Convergence Speed (Loss)** for all configurations:
```bash
python final_comparison.py
```
This generates `final_comparison_plot.png` and displays a summary table comparing best accuracies.

### 5. Error Analysis (Week 4)
Analyze misclassifications of a trained model:
```bash
python analyze_errors.py
```
This generates `superclass_confusion.png` and prints the most frequent class confusions.

## Requirements
- PyTorch / Torchvision
- Matplotlib / Seaborn
- Scikit-learn
- TQDM