# Script for analizing outputs produced by the models.

import math
from typing import Tuple
from typing import Union
from typing import List
from copy import deepcopy
import argparse
import ast
from pathlib import Path

import pandas as pd
from sympy import ShapeError
import numpy as np

from sklearn import manifold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import  classification_report
from sklearn.calibration import calibration_curve, CalibrationDisplay

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

import torch
import torchvision
import torchvision.transforms.functional as F

from PIL import ImageFont, ImageDraw, ImageOps

#Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


tab10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

#Define here the new paths
path_to_outputs = "..../examples/SL_topic_detection/outputs_final"
#MODELS = ["transformerCLS","perceiverIO"] #Add perceiver when we have it
MODELS = ["transformerCLS"]
SPLIT = ["val", "test"]
EXPERIMENT_NUM=[1,2,3,4,5]
#extra_args = ["2d_pose_handsandbody_2","2d_pose_handsandbody","2d_pose_2","2d_pose"]
extra_args = ["2d_pose_handsandbody"]
datasets=["How2Sign", "EgoSign-rgb","EgoSign-oc-homo", "EgoSign-oc-resec","EgoSign-combined-homo","EgoSign-combined-resec"] #How2Sign, EgoSign-rgb, EgoSign-oc, EgoSign-combined
data_cleaning="smooth_normalized"

def load_data_dict(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(file_path, map_location=torch.device('cpu'))
    if type(data) == type(dict()):
        return data  # ['embeddings'], data['targets'], data['preds'], data['att_time'], ['att_inputs']
    raise TypeError(f'Expected data container to be of type `{type(dict)}` but got `{type(data)}` instead.')

def plot_precision_recall(
    targets_binary: List[int],
    preds_binary: List[int],
    model: str,
    data_type: str,
    split: str,
    average: str,
    num_exp,
    ) -> None:

    precision = dict()
    recall = dict()
    average_precision = dict()
    precision[average], recall[average], _ = precision_recall_curve(
        targets_binary.ravel(), preds_binary.ravel()
    )
    average_precision[average] = average_precision_score(targets_binary, preds_binary, average=average)

    display = PrecisionRecallDisplay(
        recall=recall[average],
        precision=precision[average],
        average_precision=average_precision[average],
    )
    display.plot()
    _ = display.ax_.set_title(f"{average}-average; {model} - {data_type} - {split}")
    plt.savefig(f'{path_to_outputs}/{average}-average_precision_recall_{model}_{data_type}_{num_exp}_{split}.png')
    plt.close()

    for i in range(10):
        precision[i], recall[i], _ = precision_recall_curve(targets_binary[:, i], preds_binary[:, i])
        average_precision[i] = average_precision_score(targets_binary[:, i], preds_binary[:, i])

    _, ax = plt.subplots(figsize=(8, 8))

    for i, color in zip(range(10), tab10):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"class {i}", color=color)
    display = PrecisionRecallDisplay(
        recall=recall[average],
        precision=precision[average],
        average_precision=average_precision[average],
    )
    display.plot(ax=ax, name=f"{average}-average precision-recall", color="gold")

    handles, labels = display.ax_.get_legend_handles_labels()
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title(f'{model} - {data_type} - {split}')
    plt.savefig(f'{path_to_outputs}/{average}-average_precision_recall_multiclass_{model}_{data_type}_{num_exp}_{split}.png')
    plt.close()

def plot_confusion_matrix(
    targets: Union[List[int], torch.Tensor],
    preds: Union[List[int], torch.Tensor],
    model: str,
    data_type: str,
    split: str,
    num_exp: int,
    save_path: str,
    ) -> None:

    disp = ConfusionMatrixDisplay.from_predictions(targets, preds, cmap=plt.cm.Blues, colorbar=False)
    disp.figure_.suptitle(f'{model} - {data_type} - {split}')

    plt.savefig(save_path)
    plt.close()

def metrics_to_csv(
    targets: Union[List[int], torch.Tensor],
    preds: Union[List[int], torch.Tensor],
    model: str,
    data_type: str,
    split: str,
    ) -> None:

    report = classification_report(
        targets,
        preds,
        # labels=[i for i in range(1, 11)],
        # target_names=[i for i in range(1, 11)],
        digits=4,
        output_dict=True,
        zero_division='warn',
    )

    report = pd.DataFrame.from_dict(report, orient='columns').transpose()
    report.to_csv(f'{path_to_outputs}/metrics_report_{model}_{data_type}_{num_exp}_{split}.csv')

    support = report.pop('support')
    report, weighted_avg = report.drop(report.tail(1).index),report.tail(1)
    report, macro_avg = report.drop(report.tail(1).index),report.tail(1)
    report, accuracy = report.drop(report.tail(1).index),report.tail(1)

    report = report.append(weighted_avg)
    report = report.append(macro_avg)

    accuracy = accuracy.iloc[0,0]

    ax = report.plot.bar(
        rot=0,
        width=0.7,
        edgecolor='white',
        linewidth=1.5,
        color=["#ff7f0e", "#bcbd22", "#8c564b"],
        figsize=(11, 5),
    )
    ax.axes.set_xlim(-0.5,11.5)
    leg1 = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        fancybox=True,
        shadow=True
    )

    leg2 = ax.legend(
        [f"accuracy = " + "{:.2f}".format(accuracy*100)],
        handles=[
            Line2D(
                [0], [0], marker='o', color='w', label=f"accuracy = " + "{:.2f} %".format(accuracy*100),
                markerfacecolor='g', markersize=0)
        ],
        loc='upper center',
        bbox_to_anchor=(0.85, 1.065),
        borderaxespad=0,
        fontsize='x-large',
        frameon=False,
    )
    ax.add_artist(leg1)

    plt.xticks([i for i in range(12)], [i for i in range(1, 11)] + ['w_avg', 'macro_avg'])
    plt.savefig(f'{path_to_outputs}/metrics_barchart_{model}_{data_type}_{num_exp}_{split}.png')
    plt.close()

    ax = support.iloc[0:10].plot.bar(rot=0, width=0.7, edgecolor='white', linewidth=1, color=tab10)
    ax.set_title(f"Samples per class in {split} set")
    plt.xticks([i for i in range(10)], [i for i in range(1, 11)])
    plt.savefig(f'{path_to_outputs}/metrics_support_{model}_{data_type}_{num_exp}_{split}.png')
    plt.close()

def analysis_of_errors(
    targets: Union[List[int], torch.Tensor],
    preds: Union[List[int], torch.Tensor],
    logits: Union[List[float], torch.Tensor],
    labels: List[str],
    model: str,
    data_type: str,
    split: str,
    num_exp: int,
    ) -> None:
    from sklearn.metrics import precision_recall_fscore_support as score

    from sklearn.preprocessing import label_binarize
    # Use label_binarize to fit into a multilabel setting
    targets_binary = label_binarize(targets, classes=[i for i in range(10)])
    preds_binary = label_binarize(preds, classes=[i for i in range(10)])
    
    for average in ['micro', 'macro']:
        plot_precision_recall(
            targets_binary=targets_binary,
            preds_binary=preds_binary,
            model=model,
            data_type=data_type,
            split=split,
            average=average,
            num_exp=num_exp,
        )

    plot_confusion_matrix(
        targets=targets,
        preds=preds,
        model=model,
        data_type=data_type,
        split=split,
        num_exp=num_exp,
    )

    metrics_to_csv(
        targets=targets,
        preds=preds,
        model=model,
        data_type=data_type,
        split=split,
        num_exp=num_exp,
    )

def obtain_tSNE_projection(
    embeddings: Union[torch.Tensor, np.array],
    ) -> np.array:
    # TODO: set a grid for each of the models (3 in total),
    #       with 4 x 3 = 12 subplots each (4 data types, 3 dataset splits)
    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.numpy()
    if len(embeddings.shape) != 2:
        raise RuntimeError(
            (f'Expected input embeddings to be two-dimensional tensor'
            f' but got a `{len(embeddings.shape)}-dimensional tensor instead.`')
        )
    tsne = manifold.TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12,
        learning_rate="auto",
        n_iter=1000,
        random_state=41,
        n_jobs=-1,
    )
    Y = tsne.fit_transform(embeddings)
    return Y

def plot_projection(
    Y: np.array,
    class_labels: Union[List[int], List[str], torch.Tensor],
    labels: List[str],
    model: str,
    data_type: str,
    split: str,
    num_exp: int,
    ) -> None:
    if type(class_labels) == torch.Tensor:
        class_labels = deepcopy(class_labels).tolist()

    dpi = 100
    fig, ax = plt.subplots(figsize=(850/dpi,850/dpi), dpi=dpi)

    ax.axis('off')
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=class_labels, cmap='tab10')
    ax.set_title(f'{model} - {data_type} - {split}')
    plt.savefig(f'{path_to_outputs}/tsne_{model}_{data_type}_{num_exp}_{split}.png', bbox_inches='tight')
    plt.close()

# maps the label's number to its corresponding string e.g. 0 -> "beauty"
def label_num_to_str(
    class_labels: Union[List[int], List[str], torch.Tensor],
    mapping_file: str = '..../mediapipe_keypoints/categoryName_categoryID.csv', #TODO: maybe pass this as a parameter
    ) -> List[str]:
    if type(class_labels) == torch.Tensor:
        class_labels = deepcopy(class_labels).tolist()

    mapping = pd.read_csv(mapping_file, index_col=0, squeeze=True, sep=',').to_dict()
    assert len(mapping.keys()) == 10
    return mapping

def obtain_labels(targets):
    mapping = label_num_to_str(targets)
    labels = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
    return labels

def create_detailed_metrics_table(all_metrics: dict, model: str, data_type: str, path_to_save_final_metrics: str) -> None:
    """
    Creates and prints a detailed table of metrics for each split,
    showing individual seed performance and the final mean/std.
    """
    mean_valtest_metrics = None
    with open(path_to_save_final_metrics, 'w') as f:
        for split_name, split_metrics in all_metrics.items():
            if not split_metrics:
                continue

            # Convert the dictionary of metrics into a list of dictionaries for DataFrame creation
            records = [{'Seed': seed, **metrics} for seed, metrics in split_metrics.items()]
            df = pd.DataFrame.from_records(records)

            if df.empty:
                continue

            # Calculate Mean and Std Dev for each metric column
            means = df.drop(columns='Seed').mean()
            stds = df.drop(columns='Seed').std()

            # Create the summary row
            summary = {'Seed': 'Mean ± Std'}
            for col in means.index:
                summary[col] = f"{means[col]:.4f} ± {stds[col]:.4f}"
            
            # Append summary row using pd.concat
            summary_df = pd.DataFrame([summary])
            df_final = pd.concat([df.round(4), summary_df], ignore_index=True)

            # Print the final detailed table for the split
            print("---" * 20)
            print(f"Detailed Metrics for: {model} | {data_type} | Split: {split_name}")
            print("---" * 20)
            print(df_final.to_string(index=False))
            print("\n")
            
            #Also print in the file
            f.write("---" * 20 + "\n")
            f.write(f"Detailed Metrics for: {model} | {data_type} | Split: {split_name}\n")
            f.write("---" * 20 + "\n")
            f.write(df_final.to_string(index=False) + "\n\n")
            
            if split_name == 'valtest':
                mean_valtest_metrics = means.to_dict()
                
    print(f"Detailed metrics report saved to: {path_to_save_final_metrics}")
    return mean_valtest_metrics

def parse_config(model, data_type, data_cleaning):
    """Translates script variables into table categories."""
    architecture = "Transformer" if "transformer" in model.lower() else "Perceiver"
    
    body_parts = "Upper Body + Hands" if "handsandbody" in data_type else "Only Hands"
    
    if "smooth_normalized" in data_cleaning:
        cleaning = "Smoothed & Norm"
    elif "smooth" in data_cleaning:
        cleaning = "Smoothed"
    else:
        cleaning = "Raw"
        
    hp = 2 if data_type.endswith('_2') else 1
    
    return architecture, body_parts, cleaning, hp

def generate_latex_output(all_final_metrics):
    """Generates and prints a DataFrame with results formatted for LaTeX."""
    # Define the structure of the LaTeX table rows and columns
    
    architectures = ["Transformer", "Perceiver"]
    body_parts_options = ["Only Hands", "Upper Body + Hands"]
    #cleaning_options = ["Raw", "Smoothed", "Smoothed & Norm"]
    cleaning_options = ["Smoothed & Norm"]
    hp_options = [1, 2]
    datasets = ["How2Sign", "EgoSign-rgb", "EgoSign-oc-homo", "EgoSign-oc-resec", "EgoSign-combined-homo", "EgoSign-combined-resec"]

    # Map internal dataset names to the headers in your LaTeX table
    dataset_to_header = {
        "How2Sign": "How2Sign",
        "EgoSign-rgb": "EgoSign-RGB",
        "EgoSign-oc-homo": "EgoSign-OC (Homo)",
        "EgoSign-oc-resec": "EgoSign-OC (Resect)",
        "EgoSign-combined-homo": "Combined (Homo)",
        "EgoSign-combined-resec": "Combined (Resect)"
    }
    column_headers = [dataset_to_header[d] for d in datasets]

    # Create a pandas DataFrame to hold the results, initialized with '---'
    index_tuples = [(arch, bp, clean, hp) for arch in architectures for bp in body_parts_options for clean in cleaning_options for hp in hp_options]
    index = pd.MultiIndex.from_tuples(index_tuples, names=["Architecture", "Body Parts", "Data Cleaning", "HP"])
    df = pd.DataFrame(index=index, columns=column_headers).fillna("---")

    # Populate the DataFrame with the collected metrics
    for config, metrics in all_final_metrics.items():
        if metrics is None:
            continue
        arch, bp, clean, hp, dataset = config
        table_header = dataset_to_header.get(dataset)
        if table_header:
            f1 = metrics.get('F1-Score', 0)
            p = metrics.get('Precision', 0)
            r = metrics.get('Recall', 0)
            acc = metrics.get('Accuracy', 0)
            
            # Format the metrics into the \res{} command
            latex_string = f"\\res{{{f1:.3f}}}{{{p:.3f}}}{{{r:.3f}}}{{{acc:.3f}}}"
            df.loc[(arch, bp, clean, hp), table_header] = latex_string
            
    # Print the final formatted table to the console
    print("\n\n" + "="*90)
    print("LATEX TABLE DATA (for 'test' split)".center(90))
    print("="*90)
    print("Copy the values from the table below and paste them into your LaTeX table cells.")
    print("-" * 90)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df)
    print("="*90 + "\n")

def compute_majority_baseline(targets: torch.Tensor) -> None:
    """
    Calculates and prints performance for a hardcoded majority class baseline.
    This baseline model always predicts class '1' for every sample.

    Args:
        targets (torch.Tensor): A tensor of the true labels (ground truth).
    """
    print("\n" + "="*90)
    print("MAJORITY CLASS BASELINE CALCULATION (Hardcoded to Class 1)".center(90))
    print("="*90)

    true_labels = targets.numpy()
    majority_class = 1  # Hardcoded as per the requirement

    # 1. Generate predictions where every prediction is the majority class
    baseline_preds = np.full_like(true_labels, fill_value=majority_class)

    print(f"Baseline computed assuming the majority class is always '{majority_class}'.")
    print(f"Total samples used for this calculation: {len(true_labels)}")
    print("-" * 90)

    # 2. Calculate and print metrics using the classification_report
    report = classification_report(
        true_labels,
        baseline_preds,
        digits=4,
        zero_division=0,
    )

    print("Baseline Performance Metrics:\n")
    print(report)
    print("="*90 + "\n")
    
def main(args):
    all_final_metrics_for_latex = {}
    
    for model in MODELS:
        for data_type in extra_args:
            for dataset in datasets:
                print(f'Doing dataset: {dataset}', flush=True)
                
                # While changing this, we first need to collect the results to all the experiments.
                raw_data_all_runs = {split: {} for split in SPLIT}
                for num_exp in EXPERIMENT_NUM:
                    for split in SPLIT:
                        print(f'Analyzing outputs from model = {model}, args = {data_type}, dataset = {dataset}, split = {split}...', flush=True)
                        print(flush=True)
                        #Check if the path exists, if not exit it.
                        path_to_pt = f'{path_to_outputs}/inference_{model}_{data_type}_{num_exp}_{split}_{dataset}_{data_cleaning}.pt'
                        file_path = Path(path_to_pt)
                        if not file_path.is_file():
                            print(f"path not found: {path_to_pt}")
                            continue #So it doesn't try to load it.
                        data = load_data_dict(path_to_pt)

                        targets = data['targets'] + 1 #The +1 is that so they are between 1 and 10 instead of 0 and 9.
                        preds = data['preds'] + 1
                        logits = data['logits']
                        labels = obtain_labels(data['targets'])

                        raw_data_all_runs[split][num_exp] = {
                            'targets': targets,
                            'preds': preds,
                            'logits': data['logits'],
                            'embeddings': data['embeddings']
                        }
                        #Y = obtain_tSNE_projection(data['embeddings'])
                        #print(f'Plotting projections; model = {model}, data type = {data_type}, dataset = {dataset}, split = {split}...', flush=True)
                        #plot_projection(Y, targets, obtain_labels(data['targets']), model, data_type, split, num_exp)

                        #print(f'analysis_of_errors; model = {model}, data type = {data_type}, dataset = {dataset}, split = {split}...', flush=True)
                        #analysis_of_errors(
                        #    targets=targets,
                        #    preds=preds,
                        #    logits=logits,
                        #    labels=labels,
                        #    model=model,
                        #    data_type=data_type,
                        #    split=split,
                        #    num_exp=num_exp,
                        #)

                    print(flush=True)
                    print(f'Analyzed outputs for model = {model}, data type = {data_type}, split = {split}', flush=True)
                    print(flush=True)
                    
                #Combined val+rest raw data
                raw_data_all_runs['valtest'] = {}
                for seed in EXPERIMENT_NUM:
                    if seed in raw_data_all_runs['val'] and seed in raw_data_all_runs['test']:
                        val_data = raw_data_all_runs['val'][seed]
                        test_data = raw_data_all_runs['test'][seed]
                        comb_targets = torch.cat((val_data['targets'], test_data['targets']))
                        comb_preds = torch.cat((val_data['preds'], test_data['preds']))
                        raw_data_all_runs['valtest'][seed] = {'targets': comb_targets, 'preds': comb_preds}
                
                        #Compute the confusion matrix
                        save_path = f'{path_to_outputs}/confusion_matrix_{model}_{data_type}_{seed}_{split}_{dataset}_{data_cleaning}.png'
                        plot_confusion_matrix(
                            targets=comb_targets,
                            preds=comb_preds,
                            model=model,
                            data_type=data_type,
                            split=split,
                            num_exp=seed,
                            save_path=save_path,
                        )
                        print(f"Saved confusion matrix to: {save_path}")
                            
                #Per-seed metric calculation
                all_metrics = {}
                for split_name, runs_data in raw_data_all_runs.items():
                    all_metrics[split_name] = {}
                    for seed, data in runs_data.items():
                        report = classification_report(
                            data['targets'],
                            data['preds'],
                            digits=4,
                            output_dict=True,
                            zero_division=0,
                        )
                        # Store the metrics we care about
                        all_metrics[split_name][seed] = {
                            'Accuracy': report['accuracy'],
                            'Precision': report['macro avg']['precision'],
                            'Recall': report['macro avg']['recall'],
                            'F1-Score': report['macro avg']['f1-score']
                        }
                    
                #When we have everything, we average it.
                path_to_save_final_metrics = f'{path_to_outputs}/inference_metrics_{model}_{data_type}_{dataset}.info'
                valtest_metrics = create_detailed_metrics_table(all_metrics, model, data_type, path_to_save_final_metrics)
                #Save the table in this path
                
                # Parse the configuration
                architecture, body_parts, cleaning, hp = parse_config(model, data_type, data_cleaning)
                        
                # Store the mean metrics for later
                config_tuple = (architecture, body_parts, cleaning, hp, dataset)
                all_final_metrics_for_latex[config_tuple] = valtest_metrics #This are just the means!
                
    #Get the majority class metrics. That is, always predict "sports and fitness" class, which would be class 1.
    compute_majority_baseline(comb_targets) #this should be the same in all the runs, from that we want to compute the most probable class which is class 1.
    
    print("\n\n--- Generating final LaTeX output based on all 'valtest' splits ---")
    generate_latex_output(all_final_metrics_for_latex)            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)