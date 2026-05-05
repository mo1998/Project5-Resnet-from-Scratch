import json
import matplotlib.pyplot as plt
import os
import glob

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

# Find all result files
result_files = glob.glob('results/*.json')
all_data = {}

for fpath in result_files:
    name = os.path.basename(fpath).replace('.json', '')
    data = load_json(fpath)
    if data:
        all_data[name] = data

if not all_data:
    print("No result files found in results/ directory.")
    exit()

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for name, data in all_data.items():
    label = name.replace('resnet18_', '').replace('_', ' ').title()
    
    # Plot Accuracy
    if 'test_acc' in data:
        ax1.plot(data['test_acc'], label=f'{label} (Test Acc)', marker='o')
    
    # Plot Convergence Speed (Loss)
    if 'train_loss' in data:
        ax2.plot(data['train_loss'], label=f'{label} (Train Loss)', marker='s')

ax1.set_title('Test Accuracy Comparison')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.set_title('Convergence Speed (Training Loss)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross Entropy Loss')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('final_comparison_plot.png')
print(f"Saved comprehensive comparison plot (Accuracy & Loss) to 'final_comparison_plot.png'")

# Print a summary table
print("\n" + "="*60)
print(f"{'Experiment':<35} | {'Best Test Acc (%)':<15}")
print("-" * 60)

for name, data in all_data.items():
    label = name.replace('resnet18_', '').replace('_', ' ').title()
    if 'test_acc' in data and data['test_acc']:
        best_acc = max(data['test_acc'])
        print(f"{label:<35} | {best_acc:<15.2f}")
    else:
        print(f"{label:<35} | {'N/A':<15}")
print("="*60)
