'''
- read all the lines in file egosign_final_code/examples/SL_topic_detection/interence_transformer_val_test.out
- discard unecessary lines, only keep the lines that match this criterion:
keep lines starting with "Finishing experiment", "manifest_file"
keep lines that contains "Accuracy:"
- return the average and std for the metrics
Since we have accuracy, we have different experiments that we want to average. In the line that starts with Finishing experiment, we can see the experiment number and split. for example, herre is a line: Finishing experiment 1, val split
The average and std should be computed across experiments of the same split. Either val, train or test

'''
import os
import re
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np



folder_path = 'path_to_code/egosign_final_code/examples/SL_topic_detection/outputs_final'
architecture = 'transformer' # 'transformer, 'perceiver'

# Regular expressions
signer_metrics_re = re.compile(
    r"Signer ID: (\d+), Accuracy: ([0-9.]+), Precision: ([0-9.]+), Recall: ([0-9.]+), F1: ([0-9.]+)"
)
total_accuracy_re = re.compile(r"Accuracy:\s*([0-9.]+)$")
experiment_start_re = re.compile(r"Starting experiment (\d+), (\w+) split")

# Store data
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # [dataset][split][signer] = list of accuracies
totals = defaultdict(lambda: defaultdict(list))  # [dataset][split] = list of total accuracies

#Determine the valid files for the selected architecture.
all_files = os.listdir(folder_path)
selected_files = [f for f in all_files if f.endswith('.out') and architecture in f]
print(f"Processing {len(selected_files)} files for architecture '{architecture}'...\n")
print(f"Files: {selected_files}")

import pdb; pdb.set_trace()
dataset_order = ['how2sign', 'egosign-rgb', 'egosign-combined-homo', 'egosign-combined-resec'] #when we have hands + body
dataset_order = ['how2sign', 'egosign-rgb', 'egosign-oc-homo', 'egosign-oc-resec', 'egosign-combined-homo', 'egosign-combined-resec'] # for only hands

for filename in selected_files:
    file_path = os.path.join(folder_path, filename)

    # Infer dataset from filename
    base = filename.lower()
    dataset = (
        'how2sign' if 'how2sign' in base else
        'egosign-front' if 'front' in base else
        'egosign-oc' if 'oc' in base else
        'combined_improved' if 'improved' in base else
        'combined'
    )

    current_split = None
    seen_signers = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect split from "Starting experiment" line
            exp_match = experiment_start_re.search(line)
            if exp_match:
                current_split = exp_match.group(2).lower()
                seen_signers = set()  # reset per experiment

            signer_match = signer_metrics_re.search(line)
            if signer_match and current_split:
                signer_id = int(signer_match.group(1))
                acc = float(signer_match.group(2))
                results[dataset][current_split][signer_id].append(acc)
                seen_signers.add(signer_id)

            total_match = total_accuracy_re.search(line)
            if total_match and current_split and seen_signers:
                total_acc = float(total_match.group(1))
                totals[dataset][current_split].append(total_acc)

# === Reporting ===
for dataset in results:
    for split in results[dataset]:
        print(f"\n=== Dataset: {dataset}, Split: {split} ===")
        signer_stats = []
        for signer_id, accs in results[dataset][split].items():
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            print(f"Signer {signer_id}: Accuracy Mean = {mean_acc:.4f}, Std = {std_acc:.4f} (n={len(accs)} runs)")
            signer_stats.append((signer_id, mean_acc, std_acc))

        # Total accuracy across runs
        if totals[dataset][split]:
            accs = totals[dataset][split]
            mean_total = statistics.mean(accs)
            std_total = statistics.stdev(accs) if len(accs) > 1 else 0.0
            print(f"\n>> Total Accuracy: Mean = {mean_total:.4f}, Std = {std_total:.4f} (n={len(accs)} runs)")

# === Plotting ===
majority_class_accuracy = {
    'val': 0.26087,
    'test': 0.22069
}
for split in ['val', 'test']:
    all_signers = set()
    all_datasets = []

    # Collect all unique signer IDs and datasets for this split
    for dataset in results:
        if split in results[dataset]:
            all_datasets.append(dataset)
            all_signers.update(results[dataset][split].keys())

    if not all_datasets:
        continue  # nothing to plot for this split

    all_signers = sorted(all_signers)
    n_datasets = len(all_datasets)
    x = np.arange(len(all_signers))  # positions for each signer

    width = 0.8 / n_datasets  # bar width per dataset
    offset = -0.4 + width / 2  # center bars

    # Create plot
    plt.figure(figsize=(14, 6))
    
    dataset_colors = {}
    ordered_datasets = [d for d in dataset_order if d in all_datasets] #We do this to ensure correct order.
    for i, dataset in enumerate(ordered_datasets):
        means = []
        stds = []
        for signer_id in all_signers:
            if signer_id in results[dataset][split]:
                accs = results[dataset][split][signer_id]
                means.append(statistics.mean(accs))
                stds.append(statistics.stdev(accs) if len(accs) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

        bar_container = plt.bar(x + offset + i * width, means, width=width, label=dataset, yerr=stds, capsize=4)
        dataset_colors[dataset] = bar_container[0].get_facecolor()

    #Add the dotted lines
    #For the different datasets
    for dataset in ordered_datasets:
        dataset_mean = statistics.mean(totals[dataset][split])/100
        plt.axhline(
            y=dataset_mean,
            color=dataset_colors[dataset],
            linestyle=':',
            linewidth=2,
            label=f'{dataset} Mean'
        )
    
    #Majority class baseline
    plt.axhline(
        y=majority_class_accuracy[split],
        color='black',
        linestyle=':',
        linewidth=2,
        label='Majority Class'
    )
    
    plt.xticks(x, all_signers)
    plt.xlabel("Signer ID")
    plt.ylabel("Accuracy")
    plt.title(f"Combined Accuracy per Signer - {architecture} ({split})")
    plt.legend(title="Dataset")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/{architecture}_{split}_combined_accuracy_per_signer.png")
    plt.show()


'''
print(f"file_path: {file_path}")
# We'll track accuracies per split, with experiment number
split_experiments = defaultdict(dict)

current_split = None
current_exp = None

with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        # Detect start of new experiment
        if line.startswith("Starting experiment"):
            match = re.search(r"Starting experiment (\d+), (\w+) split, (\w+) features", line)
            if match:
                current_exp = int(match.group(1))
                current_split = match.group(2)
        elif "Accuracy:" in line and current_split and current_exp is not None:
            match = re.search(r"Accuracy:\s*([0-9.]+)", line)
            if match:
                acc = float(match.group(1))
                print(f"Found accuracy {acc} for experiment {current_exp} in {current_split} split.\n")
                # Only store one accuracy per experiment
                if current_exp not in split_experiments[current_split]:
                    split_experiments[current_split][current_exp] = acc
                else:
                    print(f"Warning: Multiple accuracy values for experiment {current_exp} in {current_split} split. Ignoring extra.")

# Compute averages
for split, exps in split_experiments.items():
    accuracies = list(exps.values())
    if accuracies:
        avg = statistics.mean(accuracies)
        std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        print(f"{split.capitalize()} split: Avg Accuracy = {avg:.2f}, Std = {std:.2f}, Values = {accuracies}")
    else:
        print(f"{split.capitalize()} split: No accuracy data.")
'''


'''
We will do this on the side:

For perceiver
test
How2Sign
2025-05-13 16:45:13 | INFO | __main__ | Signer ID: 1, Accuracy: 0.1875, Precision: 0.0833, Recall: 0.1500, F1: 0.0936
2025-05-13 16:45:13 | INFO | __main__ | Signer ID: 10, Accuracy: 0.3043, Precision: 0.2789, Recall: 0.2667, F1: 0.1879
2025-05-13 16:45:13 | INFO | __main__ | Signer ID: 2, Accuracy: 0.6667, Precision: 0.3571, Recall: 0.3167, F1: 0.3194
2025-05-13 16:45:13 | INFO | __main__ | Signer ID: 3, Accuracy: 0.3750, Precision: 0.2333, Recall: 0.2775, F1: 0.2076
2025-05-13 16:45:13 | INFO | __main__ | Signer ID: 5, Accuracy: 0.4194, Precision: 0.2905, Recall: 0.2686, F1: 0.2535
2025-05-13 16:45:13 | INFO | __main__ | Signer ID: 8, Accuracy: 0.3448, Precision: 0.1729, Recall: 0.1986, F1: 0.1779
2025-05-13 16:45:13 | WARNING | __main__ | Merging Accuracy requires CUDA.
2025-05-13 16:45:13 | INFO | __main__ | Accuracy: 37.6712

EgoSign
2025-05-13 16:46:58 | INFO | __main__ | Signer ID: 12, Accuracy: 0.3200, Precision: 0.2000, Recall: 0.1900, F1: 0.1696
2025-05-13 16:46:58 | INFO | __main__ | Signer ID: 13, Accuracy: 0.2273, Precision: 0.0822, Recall: 0.1333, F1: 0.0792
2025-05-13 16:46:58 | INFO | __main__ | Signer ID: 14, Accuracy: 0.2632, Precision: 0.2483, Recall: 0.2417, F1: 0.1721
2025-05-13 16:46:58 | INFO | __main__ | Signer ID: 15, Accuracy: 0.2632, Precision: 0.1233, Recall: 0.2167, F1: 0.1556
2025-05-13 16:46:58 | INFO | __main__ | Signer ID: 16, Accuracy: 0.4615, Precision: 0.1833, Recall: 0.2250, F1: 0.1567
2025-05-13 16:46:58 | INFO | __main__ | Signer ID: 8, Accuracy: 0.3542, Precision: 0.2009, Recall: 0.1939, F1: 0.1756
2025-05-13 16:46:58 | WARNING | __main__ | Merging Accuracy requires CUDA.
2025-05-13 16:46:58 | INFO | __main__ | Accuracy: 31.5068

EgoSign-oc
2025-05-13 16:47:36 | INFO | __main__ | Signer ID: 12, Accuracy: 0.2500, Precision: 0.1411, Recall: 0.1433, F1: 0.1295
2025-05-13 16:47:36 | INFO | __main__ | Signer ID: 13, Accuracy: 0.2273, Precision: 0.1250, Recall: 0.1333, F1: 0.0900
2025-05-13 16:47:36 | INFO | __main__ | Signer ID: 14, Accuracy: 0.3158, Precision: 0.2017, Recall: 0.2833, F1: 0.1934
2025-05-13 16:47:36 | INFO | __main__ | Signer ID: 15, Accuracy: 0.3158, Precision: 0.2139, Recall: 0.2000, F1: 0.1857
2025-05-13 16:47:36 | INFO | __main__ | Signer ID: 16, Accuracy: 0.2308, Precision: 0.0375, Recall: 0.0750, F1: 0.0500
2025-05-13 16:47:36 | INFO | __main__ | Signer ID: 8, Accuracy: 0.2083, Precision: 0.0981, Recall: 0.1386, F1: 0.0956
2025-05-13 16:47:36 | WARNING | __main__ | Merging Accuracy requires CUDA.
2025-05-13 16:47:36 | INFO | __main__ | Accuracy: 24.8276

val
How2Sign
2025-05-13 16:48:28 | INFO | __main__ | Signer ID: 1, Accuracy: 0.2941, Precision: 0.2750, Recall: 0.1783, F1: 0.1567
2025-05-13 16:48:28 | INFO | __main__ | Signer ID: 2, Accuracy: 0.5294, Precision: 0.3369, Recall: 0.3267, F1: 0.3133
2025-05-13 16:48:28 | INFO | __main__ | Signer ID: 3, Accuracy: 0.2692, Precision: 0.1523, Recall: 0.1383, F1: 0.1436
2025-05-13 16:48:28 | INFO | __main__ | Signer ID: 5, Accuracy: 0.3478, Precision: 0.1611, Recall: 0.2533, F1: 0.1935
2025-05-13 16:48:28 | INFO | __main__ | Signer ID: 8, Accuracy: 0.3125, Precision: 0.2450, Recall: 0.2700, F1: 0.2119
2025-05-13 16:48:28 | WARNING | __main__ | Merging Accuracy requires CUDA.
2025-05-13 16:48:28 | INFO | __main__ | Accuracy: 33.9130

EgoSign
2025-05-13 16:49:06 | INFO | __main__ | Signer ID: 12, Accuracy: 0.3636, Precision: 0.2571, Recall: 0.1532, F1: 0.1612
2025-05-13 16:49:06 | INFO | __main__ | Signer ID: 14, Accuracy: 0.2222, Precision: 0.0444, Recall: 0.1200, F1: 0.0643
2025-05-13 16:49:06 | INFO | __main__ | Signer ID: 15, Accuracy: 0.1852, Precision: 0.0619, Recall: 0.0929, F1: 0.0739
2025-05-13 16:49:06 | INFO | __main__ | Signer ID: 16, Accuracy: 0.3182, Precision: 0.1108, Recall: 0.2095, F1: 0.1264
2025-05-13 16:49:06 | INFO | __main__ | Signer ID: 8, Accuracy: 0.6667, Precision: 0.2167, Recall: 0.3000, F1: 0.2467
2025-05-13 16:49:06 | WARNING | __main__ | Merging Accuracy requires CUDA.
2025-05-13 16:49:06 | INFO | __main__ | Accuracy: 29.5652

EgoSign-oc
2025-05-13 16:49:33 | INFO | __main__ | Signer ID: 12, Accuracy: 0.4545, Precision: 0.1787, Recall: 0.2347, F1: 0.1934
2025-05-13 16:49:33 | INFO | __main__ | Signer ID: 14, Accuracy: 0.1852, Precision: 0.0893, Recall: 0.1133, F1: 0.0920
2025-05-13 16:49:33 | INFO | __main__ | Signer ID: 15, Accuracy: 0.1481, Precision: 0.0625, Recall: 0.0679, F1: 0.0650
2025-05-13 16:49:33 | INFO | __main__ | Signer ID: 16, Accuracy: 0.2727, Precision: 0.0694, Recall: 0.1238, F1: 0.0864
2025-05-13 16:49:33 | INFO | __main__ | Signer ID: 8, Accuracy: 0.5000, Precision: 0.1667, Recall: 0.2000, F1: 0.1800
2025-05-13 16:49:33 | WARNING | __main__ | Merging Accuracy requires CUDA.
2025-05-13 16:49:33 | INFO | __main__ | Accuracy: 28.6957

For transformer




import matplotlib.pyplot as plt

# Data input: metrics by dataset and split
# Format: {split: {dataset: [(signer_id, acc, prec, recall, f1), ...]}}

data = {
    'test': {
        'How2Sign': [
            (1, 0.1875, 0.0833, 0.1500, 0.0936),
            (2, 0.6667, 0.3571, 0.3167, 0.3194),
            (3, 0.3750, 0.2333, 0.2775, 0.2076),
            (5, 0.4194, 0.2905, 0.2686, 0.2535),
            (8, 0.3448, 0.1729, 0.1986, 0.1779),
            (10, 0.3043, 0.2789, 0.2667, 0.1879)
        ],
        'EgoSign': [
            (8, 0.3542, 0.2009, 0.1939, 0.1756),
            (12, 0.3200, 0.2000, 0.1900, 0.1696),
            (13, 0.2273, 0.0822, 0.1333, 0.0792),
            (14, 0.2632, 0.2483, 0.2417, 0.1721),
            (15, 0.2632, 0.1233, 0.2167, 0.1556),
            (16, 0.4615, 0.1833, 0.2250, 0.1567)
        ],
        'EgoSign-oc': [
            (8, 0.2083, 0.0981, 0.1386, 0.0956),
            (12, 0.2500, 0.1411, 0.1433, 0.1295),
            (13, 0.2273, 0.1250, 0.1333, 0.0900),
            (14, 0.3158, 0.2017, 0.2833, 0.1934),
            (15, 0.3158, 0.2139, 0.2000, 0.1857),
            (16, 0.2308, 0.0375, 0.0750, 0.0500)
        ],
    },
    'val': {
        'How2Sign': [
            (1, 0.2941, 0.2750, 0.1783, 0.1567),
            (2, 0.5294, 0.3369, 0.3267, 0.3133),
            (3, 0.2692, 0.1523, 0.1383, 0.1436),
            (5, 0.3478, 0.1611, 0.2533, 0.1935),
            (8, 0.3125, 0.2450, 0.2700, 0.2119),
        ],
        'EgoSign': [
            (8, 0.6667, 0.2167, 0.3000, 0.2467),
            (12, 0.3636, 0.2571, 0.1532, 0.1612),
            (14, 0.2222, 0.0444, 0.1200, 0.0643),
            (15, 0.1852, 0.0619, 0.0929, 0.0739),
            (16, 0.3182, 0.1108, 0.2095, 0.1264),
        ],
        'EgoSign-oc': [
            (8, 0.5000, 0.1667, 0.2000, 0.1800),
            (12, 0.4545, 0.1787, 0.2347, 0.1934),
            (14, 0.1852, 0.0893, 0.1133, 0.0920),
            (15, 0.1481, 0.0625, 0.0679, 0.0650),
            (16, 0.2727, 0.0694, 0.1238, 0.0864),
        ],
    }
}

colors = {
    'How2Sign': 'blue',
    'EgoSign': 'green',
    'EgoSign-oc': 'orange'
}

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
metric_indices = {
    'Accuracy': 1,
    'Precision': 2,
    'Recall': 3,
    'F1': 4
}

# Plotting function
def plot_metrics(split_name, split_data):
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        for dataset, entries in split_data.items():
            signer_ids = [entry[0] for entry in entries]
            metric_vals = [entry[metric_indices[metric]] for entry in entries]
            plt.plot(signer_ids, metric_vals, marker='o', label=dataset, color=colors[dataset])
        plt.title(f'{metric} per Signer - {split_name.upper()}')
        plt.xlabel('Signer ID')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'inference_perceiver_{split_name}_{metric}.png')

# Plot both test and val
for split_name, split_data in data.items():
    plot_metrics(split_name, split_data)


'''