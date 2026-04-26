# ResNet from Scratch vs. Transfer Learning on CIFAR-100

This project explores the implementation of ResNet-18, the impact of Batch Normalization, data augmentation, and a comparison with transfer learning.

## Project Structure

- `models.py`: ResNet-18 implementation from scratch.
- `train.py`: Basic training script for the scratch model.
- `compare_bn.py`: Experiment comparing training with and without Batch Normalization (Week 1 & 2). Supports toggling augmentation and setting epochs.
- `transfer_learning.py`: Script to fine-tune a pretrained ResNet-18 on CIFAR-100 (Week 2). Logs both accuracy and training loss.
- `train_advanced.py`: Advanced training with Cutout, Mixup, and Cosine Annealing (Week 3).
- `analyze_errors.py`: Detailed error analysis at fine-class and superclass levels (Week 4).
- `final_comparison.py`: Automatically aggregates all JSON results from `results/` to generate a dual-plot comparison (Accuracy and Convergence Speed).
- `results/`: Directory containing JSON logs of experiment metrics.