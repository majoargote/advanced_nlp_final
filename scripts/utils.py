
import re

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Metrics class 
class Metrics:
    
    def __init__(self):
        self.results = {}

    def run(self, y_true, y_pred, method_name, average='macro'):
        self.results[method_name] = {
            "Accuracy":  accuracy_score(y_true, y_pred) * 100,
            "Precision": precision_score(y_true, y_pred, average=average) * 100,
            "Recall":    recall_score(y_true, y_pred, average=average) * 100,
            "F1-Score":  f1_score(y_true, y_pred, average=average) * 100
        }

    def plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        method_names  = list(self.results.keys())
        colors = plt.cm.tab10.colors[:len(method_names)]  # works for up to 10 methods

        for i, metric in enumerate(metrics_names):
            ax = axes[i // 2, i % 2]
            metric_values = [self.results[m][metric] for m in method_names]

            bars = ax.bar(method_names, metric_values, color=colors)
            ax.set_title(metric)
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)  # 45-degree x-axis labels

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                        f'{height:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        # return fig


## get most common words in reviews for each category

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import string

STOPWORDS_ADDITIONAL = set(['would',
 'service',
 'time',
 'good',
 'delivery',
 'get',
 'order',
 'one',
 'great',
 'company',
 'customer',
 'day',
 'really',
 'use',
 'back',
 'received',
 'ordered',
 'could',
 'still',
 'days',
 'like',
 'experience',
 'also',
 'even',
 'well',
 'got',
 'email',
 'arrived',
 'first',
 'money',
 'helpful',
 '2',
 'made',
 'told',
 'recommend',
 'us',
 'easy',
 'however',
 'sent',
 'never',
 'new',
 'much',
 'always',
 'work',
 'phone',
 'quality',
 'product',
 'price',
 'said',
 'refund'])

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english')).union(STOPWORDS_ADDITIONAL)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def preprocess_for_tfidf(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)  # return string for TF-IDF vectorizer

def preprocess_for_bert(text):
    text = str(text).strip()
    text = re.sub(r'<.*?>', '', text)   # remove HTML tags (noise)
    text = re.sub(r'\s+', ' ', text)    # normalize whitespace
    return text


### Metrics for compressing model evaluation

import time
import psutil
import os
import gc
import logging
from codecarbon import EmissionsTracker
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logging.getLogger("codecarbon").disabled = True

# Improved helper function to measure inference speed and memory usage
def measure_inference_metrics(model, dataset, device="cpu", batch_size=32):
    model.to(device)
    model.eval()
    
    # Clear memory and get baseline measurements
    gc.collect()
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gpu_memory_baseline = torch.cuda.memory_allocated(device) / 1e6  # MB
    
    process = psutil.Process(os.getpid())
    cpu_memory_baseline = process.memory_info().rss / 1e6  # MB
    
    # Start tracking
    tracker = EmissionsTracker(project_name="bert_agnews", measure_power_secs=1)
    tracker.start()
    
    start_time = time.time()
    total_samples = len(dataset)
    
    # Collect all predictions and true labels for macro metrics
    all_predictions = []
    all_labels = []
    
    # Track peak memory during inference
    peak_cpu_memory = cpu_memory_baseline
    
    # Evaluate in batches
    for i in tqdm(range(0, total_samples, batch_size)):
        batch = dataset[i: i + batch_size]
        inputs = {
            "input_ids": batch["input_ids"].to(device, dtype=torch.long),
            "attention_mask": batch["attention_mask"].to(device, dtype=torch.long)
        }
        labels = batch["label"].to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Handle different output formats from compressed models
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                logits = outputs[0]
            else:
                raise ValueError(f"Unexpected output format: {type(outputs)}")
            
            preds = torch.argmax(logits, axis=-1)
            
            # Collect predictions and labels for metric calculation
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Track peak CPU memory during inference
        current_cpu_memory = process.memory_info().rss / 1e6
        peak_cpu_memory = max(peak_cpu_memory, current_cpu_memory)
    
    end_time = time.time()
    emissions: float = tracker.stop()
    
    # Calculate macro metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    precision_macro = precision_score(all_labels, all_predictions, average='macro')
    recall_macro = recall_score(all_labels, all_predictions, average='macro')
    
    # Final memory measurements
    cpu_memory_final = process.memory_info().rss / 1e6  # MB
    cpu_memory_used = peak_cpu_memory - cpu_memory_baseline
    
    metrics = {
        "inference_speed (samples/sec)": total_samples / (end_time - start_time),
        "cpu_memory_used (MB)": cpu_memory_used,
        "cpu_memory_peak (MB)": peak_cpu_memory,
        "carbon_footprint (kg CO2eq)": emissions,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        # Per-sample predictions/labels from the forward pass above, so callers can
        # reuse them (e.g. confusion matrices) without re-running inference.
        # Native python ints keep the dict JSON-serializable; strip before dumping if undesired.
        "predictions": [int(p) for p in all_predictions],
        "labels": [int(l) for l in all_labels],
    }
    
    # Add GPU memory metrics if using CUDA
    if torch.cuda.is_available() and device != "cpu":
        gpu_memory_peak = torch.cuda.max_memory_allocated(device) / 1e6  # MB
        gpu_memory_current = torch.cuda.memory_allocated(device) / 1e6  # MB
        gpu_memory_used = gpu_memory_peak - gpu_memory_baseline
        
        metrics.update({
            "gpu_memory_used (MB)": gpu_memory_used,
            "gpu_memory_peak (MB)": gpu_memory_peak,
            "gpu_memory_current (MB)": gpu_memory_current
        })
    
    return metrics