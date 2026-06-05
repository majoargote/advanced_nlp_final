
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