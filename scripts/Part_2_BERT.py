"""
Part 2: BERT
a. BERT Model with Limited Data (0.5 points):
Train a BERT-based model using only 32 labeled examples and assess its performance.

"""
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from utils import Metrics, preprocess_for_tfidf, preprocess_for_bert
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from datasets import Dataset as HFDataset

def set_seed(seed=123):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

SEED = 123
set_seed(SEED)

metrics_val = Metrics()

## Model/tokenizer config
# checkpoint = 'distilbert-base-uncased' #  (smaller, faster)
checkpoint = 'bert-base-uncased' # (classic BERT)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
max_length = 64


def to_hf_dataset(texts, labels=None):
    data = {'review': list(texts)}
    if labels is not None:
        data['labels'] = list(labels)
    return HFDataset.from_dict(data)

def tokenize(batch):
    return tokenizer(batch['review'], truncation=True, max_length=max_length)


def train_bert(train_texts, train_labels, valid_texts, valid_labels,
               num_train_epochs=20, freeze_encoder=False, output_dir='./tmp_bert', num_labels=5):
    
    train_ds = to_hf_dataset(train_texts, train_labels)
    valid_ds = to_hf_dataset(valid_texts, valid_labels)  # now passing labels

    train_ds = train_ds.map(tokenize, batched=True)
    valid_ds = valid_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels
    )

    if freeze_encoder and hasattr(model, 'bert'):
        for param in model.bert.parameters():
            param.requires_grad = False

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,      # restores best checkpoint for predictions
        save_total_limit=1,               # only keeps 1 checkpoint on disk at a time
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        logging_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,       # needed for early stopping
        data_collator=DataCollatorWithPadding(tokenizer),
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        # stops if val loss doesn't improve for 5 consecutive epochs
    )

    trainer.train()

    preds = trainer.predict(valid_ds)
    valid_preds = preds.predictions.argmax(-1)

    # Clean up checkpoints immediately after predictions - clean up disk space
    shutil.rmtree(output_dir, ignore_errors=True)

    return model, preds, valid_preds

def print_balance(labels, name):
    counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    print(f"\n--- {name} ({total} samples) ---")
    for cls, count in counts.items():
        print(f"  Class {cls+1} ({count/total*100:.1f}%): {'█' * count} ({count})")

def main():
    # Load data
    df = pd.read_csv("../data/filtered_reviews.csv")

    df['review'] = df['review'].apply(preprocess_for_bert)

    print(f"Full dataset: {df.shape}")
    print(df['stars'].value_counts(dropna=False))

    # Only keep rows that have a label
    labeled_df = df.dropna(subset=['stars', 'review'])
    texts = labeled_df['review'].tolist()
    labels = (labeled_df['stars'].astype(int) - 1).tolist()
    # model expects 0-4 labels for 5 classes

    # Single train/valid split — not enough data for 3 splits
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    print_balance(train_labels, "Training set")
    print_balance(valid_labels, "Validation set")

    # Apply only to TF-IDF inputs
    train_texts_tfidf = [preprocess_for_tfidf(t) for t in train_texts]
    valid_texts_tfidf = [preprocess_for_tfidf(t) for t in valid_texts]

    # TF-IDF baseline (trained on all labeled training data)
    tfidf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    tfidf_pipeline.fit(train_texts_tfidf, train_labels)
    valid_preds_tfidf = tfidf_pipeline.predict(valid_texts_tfidf)
    metrics_val.run(valid_labels, valid_preds_tfidf, 'TF-IDF (baseline)')

    # BERT limited-data experiment — use at most 32 examples
    num_labels = len(set(train_labels))
    limited_n = min(32, len(train_texts))  # cap at available data
    try:
        small_texts, _, small_labels, _ = train_test_split(
            train_texts, train_labels, train_size=limited_n, stratify=train_labels, random_state=SEED
        )
    except Exception:
        small_texts = train_texts[:limited_n]
        small_labels = train_labels[:limited_n]

    print_balance(small_labels, "BERT training set (limited)")


    # _, _, valid_preds_bert = train_bert(
    #     small_texts, small_labels, valid_texts, valid_labels,
    #     num_train_epochs=20,
    #     num_labels=num_labels,
    #     freeze_encoder=True  # already in your function signature
    # )

    _, _, valid_preds_bert = train_bert(
        small_texts, small_labels, valid_texts, valid_labels,
        num_train_epochs=20,
        num_labels=num_labels

        # defaults: 20 epochs, early stopping patience=3, lr=5e-5
    )
    metrics_val.run(valid_labels, valid_preds_bert, 'BERT (limited)')


if __name__ == "__main__":
    main()
    metrics_val.plot()


"""
I tried reducing the learning rate, adjusting the weight_decay and warm up ratio, and adding early stopping and freeze_encoder. 

learning rate: controls how much the model updates its weights in response to the estimated error each time the model weights are updated. 
A smaller learning rate can help prevent overfitting on tiny data by making more gradual updates, but it may require more epochs to converge.
weight_decay: adds L2 regularization to the loss function, which can help prevent overfitting by penalizing large weights.
warmup_ratio: gradually increases the learning rate from a small value to the target learning rate over the initial portion of training. This can help stabilize training in the early stages, especially with limited data.
freeze_encoder: prevents the pre-trained BERT encoder layers from updating during training, which can help preserve the general language understanding learned from large corpora and reduce overfitting on small datasets.

All these did not improve the F1 score for the BERT model, which suggests that the model is struggling to learn meaningful patterns from such a small dataset. 

So I reverted make to default settings that ran best with the cleanest version set up: at 20 epochs, no early stop, no freezing, lr=5e-5, no weight decay or warm up ratio

For code: I used the class example from class 8 and website notes. I then used Claude to clean my code. 
Claude also suggested things to adjust for the tiny dataset size, such as freezing the encoder and adding early stopping which I also saw in class, but these did not improve performance in my experiments.

NOTE: I will review the class notes again to ensure I tested all the recommended adjustments.

The last run showed F1-score of 49.3% for the Bert model.

With only 7 examples for validation set the F1 score is very unstable.  One single prediction changes the score by 14%.

"""