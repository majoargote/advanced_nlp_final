"""
Rule-based (regex / lexicon) baseline for predicting EXACT 1-5 star ratings
from review text, within a single chosen category.

Pipeline
--------
1. Tokenize the review (title + review).
2. Score each token against a sentiment lexicon (positive = +, negative = -),
   with stronger words given larger weights.
3. Handle simple negation: a negator ("not", "never", "no", "didn't", ...)
   flips the sign of the next few sentiment tokens.
4. Handle intensifiers ("very", "really", "extremely") which scale the next
   sentiment token, and "exclamation"/all-caps as a mild intensity bump.
5. Normalize the summed score by review length -> a polarity score.
6. Map polarity to a 1-5 star prediction via thresholds. Thresholds can be
   FIXED or auto-calibrated so predicted star frequencies match the observed
   star distribution (recommended, since lexicon scores have no natural scale).

Why exact 1-5 is hard for rules
--------------------------------
Lexicons capture POLARITY well but INTENSITY poorly. "Good" vs "amazing" is
learnable; "4 stars happy" vs "5 stars happy" usually is not expressed in the
words. Expect most error to be +/- 1 star (adjacent classes). That's why we
report MAE and quadratic-weighted kappa, not just accuracy.
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared project Metrics class (same one the whole team uses)
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self.results = {}

    def run(self, y_true, y_pred, method_name, average="macro"):
        self.results[method_name] = {
            "Accuracy": accuracy_score(y_true, y_pred) * 100,
            "Precision": precision_score(y_true, y_pred, average=average) * 100,
            "Recall": recall_score(y_true, y_pred, average=average) * 100,
            "F1-Score": f1_score(y_true, y_pred, average=average) * 100,
        }

    def plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        method_names = list(self.results.keys())
        colors = plt.cm.tab10.colors[: len(method_names)]

        for i, metric in enumerate(metrics_names):
            ax = axes[i // 2, i % 2]
            metric_values = [self.results[m][metric] for m in method_names]
            bars = ax.bar(method_names, metric_values, color=colors)
            ax.set_title(metric)
            ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                )
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 1. Sentiment lexicon
# ---------------------------------------------------------------------------
# Weights are rough intensities. Tune freely. Keep lowercase.
POSITIVE = {
    "excellent": 2.0,
    "amazing": 2.0,
    "outstanding": 2.0,
    "perfect": 2.0,
    "fantastic": 2.0,
    "superb": 2.0,
    "brilliant": 2.0,
    "wonderful": 2.0,
    "love": 1.8,
    "loved": 1.8,
    "best": 1.8,
    "exceptional": 2.0,
    "great": 1.5,
    "happy": 1.3,
    "recommend": 1.3,
    "recommended": 1.3,
    "pleased": 1.3,
    "impressed": 1.4,
    "reliable": 1.2,
    "helpful": 1.2,
    "friendly": 1.1,
    "good": 1.0,
    "nice": 1.0,
    "quick": 0.8,
    "prompt": 0.9,
    "easy": 0.8,
    "smooth": 0.9,
    "quality": 0.8,
    "satisfied": 1.2,
    "fast": 0.8,
    "efficient": 1.0,
    "fine": 0.5,
    "ok": 0.3,
    "okay": 0.3,
    "decent": 0.5,
    "thank": 0.8,
    "thanks": 0.8,
}
NEGATIVE = {
    "terrible": 2.0,
    "awful": 2.0,
    "horrible": 2.0,
    "worst": 2.0,
    "disgusting": 2.0,
    "appalling": 2.0,
    "useless": 1.8,
    "scam": 2.0,
    "fraud": 2.0,
    "rubbish": 1.7,
    "hate": 1.8,
    "hated": 1.8,
    "disappointed": 1.5,
    "disappointing": 1.5,
    "poor": 1.4,
    "bad": 1.3,
    "rude": 1.5,
    "slow": 1.0,
    "broken": 1.3,
    "faulty": 1.3,
    "refund": 0.8,
    "avoid": 1.6,
    "never": 0.5,
    "unhelpful": 1.4,
    "ignored": 1.3,
    "cancelled": 0.9,
    "delay": 1.0,
    "delayed": 1.0,
    "waste": 1.6,
    "overpriced": 1.2,
    "expensive": 0.6,
    "wrong": 1.0,
    "problem": 0.8,
    "issue": 0.7,
    "complaint": 1.0,
    "nightmare": 1.8,
    "fail": 1.4,
    "failed": 1.4,
    "unacceptable": 1.7,
    "mislead": 1.6,
    "misleading": 1.6,
}

NEGATORS = {
    "not",
    "no",
    "never",
    "nobody",
    "nothing",
    "neither",
    "nor",
    "cannot",
    "cant",
    "didnt",
    "doesnt",
    "dont",
    "isnt",
    "wasnt",
    "wouldnt",
    "couldnt",
    "wont",
    "without",
    "lack",
    "hardly",
}
INTENSIFIERS = {
    "very": 1.5,
    "really": 1.4,
    "extremely": 1.8,
    "so": 1.3,
    "absolutely": 1.6,
    "totally": 1.4,
    "super": 1.4,
    "incredibly": 1.7,
    "completely": 1.5,
    "highly": 1.5,
}

TOKEN_RE = re.compile(r"[a-z']+")


def polarity(text):
    """Return a length-normalized polarity score for one review."""
    raw = text.lower()
    exclaim = raw.count("!")
    tokens = TOKEN_RE.findall(raw)

    score = 0.0
    negate = 0  # countdown: tokens still under negation scope
    intens = 1.0  # multiplier for the next sentiment token
    for tok in tokens:
        if tok in NEGATORS:
            negate = 3  # negate next up-to-3 sentiment-bearing words
            continue
        if tok in INTENSIFIERS:
            intens = INTENSIFIERS[tok]
            continue

        val = 0.0
        if tok in POSITIVE:
            val = POSITIVE[tok]
        elif tok in NEGATIVE:
            val = -NEGATIVE[tok]

        if val != 0.0:
            val *= intens
            if negate > 0:
                val = -val
            score += val
            intens = 1.0
            negate = max(0, negate - 1)
        elif negate > 0:
            negate -= 1  # decay scope across non-sentiment words

    # mild bump from exclamation marks in the direction of current sign
    if exclaim:
        score += np.sign(score) * min(exclaim, 3) * 0.2

    # length-normalize so long reviews don't dominate
    n = max(len(tokens), 1)
    return score / np.sqrt(n)


# ---------------------------------------------------------------------------
# 2. Map polarity -> 1..5 stars
# ---------------------------------------------------------------------------
def fixed_thresholds(scores):
    """Hand-set cut points on the normalized polarity score."""
    cuts = [-0.30, -0.05, 0.10, 0.40]  # -> 1 | 2 | 3 | 4 | 5
    return np.digitize(scores, cuts) + 1


def calibrated_thresholds(scores, y_true):
    """
    Choose cut points so predicted star frequencies match the TRUE star
    distribution (quantile matching). This removes the arbitrary-scale problem
    and is the fair way to give a lexicon baseline its best shot.
    """
    scores = np.asarray(scores)
    star_counts = pd.Series(y_true).value_counts().sort_index()
    props = star_counts / star_counts.sum()
    # cumulative proportions give the quantile boundaries between classes
    qs = props.cumsum().values[:-1]  # 4 boundaries for 5 classes
    cuts = np.quantile(scores, qs)
    return np.digitize(scores, cuts) + 1


# ---------------------------------------------------------------------------
# 3. Evaluate (uses the shared Metrics class)
# ---------------------------------------------------------------------------
def predict_stars(df, category, text_fields=("title", "review"), calibrate=True):
    """Return (y_true, y_pred) for one category."""
    sub = df[df["category"] == category].copy()
    if sub.empty:
        raise ValueError(f"No rows for category {category!r}")

    text = sub[list(text_fields)].fillna("").astype(str).agg(" ".join, axis=1)
    scores = text.map(polarity).to_numpy()
    y_true = sub["stars"].to_numpy()

    if calibrate:
        y_pred = calibrated_thresholds(scores, y_true)
    else:
        y_pred = fixed_thresholds(scores)
    return y_true, y_pred


if __name__ == "__main__":
    df = pd.read_pickle(
        "/Users/camilanunezrodriguez/Documents/GitHub/advanced_nlp_final/data/trustpilot_reviews.pkl"
    )

    # Pick your two categories. Thresholds are calibrated per-category
    # (each category gets its own cut points), which is fairer when the two
    # differ in how positive/negative their reviews skew.
    CATEGORIES = ["Media & Publishing", "Travel & Vacation"]  # categories

    metrics = Metrics()
    for cat in CATEGORIES:
        y_true, y_pred = predict_stars(df, cat, calibrate=True)
        metrics.run(y_true, y_pred, method_name=f"Rule-based ({cat})")
        print(f"{'=' * 70}\n{cat}  (n = {len(y_true)})\n{'=' * 70}")
        print(classification_report(y_true, y_pred, zero_division=0))

    # comparison table + bar charts across the two categories
    print(pd.DataFrame(metrics.results).T.round(1).to_string())
    metrics.plot()
