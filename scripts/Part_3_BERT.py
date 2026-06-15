"""
Part 3
a. Full Dataset Training (0.25 points):
Incrementally train your model with varying percentages of the full dataset (1%, 10%, 25%, 50%, 75%,
and 100%). Record the results.

b. Learning Curve (0.25 points):
Plot a learning curve based on the training data percentages.

"""


from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from scripts.Part_2_BERT import train_bert
from scripts.utils import Metrics, preprocess_for_tfidf, preprocess_for_bert
import time

## set up for timing and reproducibility
def format_time(seconds):
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"

total_start = time.time()

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 123
set_seed(SEED)


## Initialize metrics and training percentages
metrics_val = Metrics()
training_percentages = [1, 10, 25, 50, 75, 100]

def plot_learning_curve(percentages, metric_results):
    """Plot the learning curve based on the training data percentages.
    Args:
        percentages (list): List of training data percentages.
        metric_results (dict): Dictionary containing F1-Score and Accuracy for each percentage.

    I choose to plot both F1-Score and Accuracy on the same graph for a comprehensive view of model performance.
    I choose to show the Accuracy and F1-Score on the same graph to provide a comprehensive view of model performance across different training data percentages.
    Accuracy captures the overall correctness of the model, while F1-Score provides insight into the balance between precision and recall, especially important in imbalanced datasets.
    While our dataset seems overall balanced, the smaller percentages may introduce some imbalance, making F1-Score a valuable metric to consider alongside Accuracy.
    By correctness of the model I mean the overall ability of the model to correctly classify the reviews into their respective star ratings. 
    Accuracy is a straightforward metric that indicates the proportion of correct predictions out of all predictions made by the model.    
    """
    ordered_percentages = sorted(percentages)
    f1_scores = [metric_results[f'BERT_{percentage}%']['F1-Score'] for percentage in ordered_percentages]
    accuracy_scores = [metric_results[f'BERT_{percentage}%']['Accuracy'] for percentage in ordered_percentages]

    plt.figure(figsize=(9, 6))
    plt.plot(ordered_percentages, f1_scores, marker='o', linewidth=2.5, label='F1-Score')
    plt.plot(ordered_percentages, accuracy_scores, marker='s', linewidth=2, linestyle='--', label='Accuracy')
    plt.title('BERT Learning Curve by Training Data Percentage')
    plt.xlabel('Training Data Percentage')
    plt.ylabel('Validation Score (%)')
    plt.xticks(ordered_percentages)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figures_dir = Path(__file__).resolve().parents[1] / 'figures'
    figures_dir.mkdir(exist_ok=True)
    output_path = figures_dir / 'part3_bert_learning_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved learning curve to {output_path}")

def print_balance(labels, name):
    """Print the class distribution for a given set of labels.
    Choosing to print the class distribution helps in understanding the balance of the dataset, especially when working with smaller percentages of data.
    It provides insight into whether the model is being trained on a representative sample of the data or if certain classes are underrepresented, which could affect model performance.
    """
    counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    print(f"\n--- {name} ({total} samples) ---")
    for cls, count in counts.items():
        ## normalize the bar so that the longest bar is 20 characters long for better visualization
        bar_length = int((count / total) * 20)
        bar = '█' * bar_length
        print(f"  Class {cls+1} ({count/total*100:.1f}%): {bar} ({count})")


"""Here I will incrementally train the BERT model with varying percentages of the full dataset (1%, 10%, 25%, 50%, 75%, and 100%).
I will 
- record the results for each percentage, including metrics such as F1-Score and Accuracy
- plot a learning curve based on the training data percentages to visualize how the model's performance changes as more data is used for training.
- print the class distribution for each training percentage to ensure that the model is being trained on a representative sample of the data and to check for any potential class imbalance issues that could affect model performance.
"""
for percentage in training_percentages:
    set_seed(SEED)  # reset at start of each iteration
    iter_start = time.time()
    print("==================================="*2)
    print(f"\nProcessing {percentage}% of data...")
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    df = pd.read_csv(data_dir / f"filtered_reviews_{percentage}percent.csv")
    df['review'] = df['review'].apply(preprocess_for_bert)
    
    print(f"Full dataset: {df.shape}")
    print(df['stars'].value_counts(dropna=False))

    # Only keep rows that have a label
    labeled_df = df.dropna(subset=['stars', 'review'])
    print(f"Labeled rows: {labeled_df.shape[0]}")
    texts = labeled_df['review'].tolist()
    labels = (labeled_df['stars'].astype(int) - 1).tolist()
    # model expects 0-4 labels for 5 classes

    # Single train/valid split — not enough data for 3 splits
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    print_balance(train_labels, "Training set")
    print_balance(valid_labels, "Validation set")   
    
    num_labels = 5 
    
    use_early_stopping = len(train_texts) > 500  # only use when enough data
    
    # Train on the full training split for this percentage
    _, _, valid_preds_bert = train_bert(
        train_texts, train_labels, valid_texts, valid_labels,
        num_train_epochs=10, # lower epochs for larger percentages to save time
        num_labels=num_labels,
        early_stopping=use_early_stopping,
        output_dir=f'./tmp_bert_{percentage}'  # unique per iteration
    )

    metrics_val.run(valid_labels, valid_preds_bert, f'BERT_{percentage}%')
    iter_elapsed = time.time() - iter_start
    total_elapsed = time.time() - total_start
    print(f"  ✓ {percentage}% done in {format_time(iter_elapsed)} | Total elapsed: {format_time(total_elapsed)}")

## plot learning curve
plot_learning_curve(training_percentages, metrics_val.results)
total_elapsed = time.time() - total_start
print(f"\nTotal runtime: {format_time(total_elapsed)}")
