"""
Part e — Optimal Technique Application, REAL SetFit version
==========================================================
Uses the actual `setfit` package (as in class) — SetFitModel + Trainer with
contrastive fine-tuning of the sentence-transformer body, then a classifier
head. This is the full SetFit method, not the frozen-embeddings shortcut.

setfit >= 1.1 fixed the DatasetFilter / eval_strategy deprecation crashes, so it
works with current transformers / huggingface_hub. NO downgrades or isolated
environment needed — just make sure you have a recent setfit:

  uv pip install --upgrade "setfit>=1.1"
  # restart the kernel if running in a notebook, then:
  cd scripts
  python part_e_setfit_real.py

Four-way data comparison, same 7 real validation reviews as parts a-d:
  (1) real only
  (2) real + part-b augmented
  (3) real + part-d synthetic
  (4) real + both
Metrics via the shared utils.Metrics class.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from datasets import Dataset

from setfit import SetFitModel, Trainer, TrainingArguments  # modern API (>=1.1)

from utils import Metrics, preprocess_for_bert

# ----------------------------------------------------------------------------
REAL_CSV = "/Users/camilanunezrodriguez/Documents/GitHub/advanced_nlp_final/data/filtered_reviews.csv"
SYNTH_CSV = "/Users/camilanunezrodriguez/Documents/GitHub/advanced_nlp_final/scripts/synthetic_filtered.csv"  # part d (LLM-generated)
AUG_CSV = "/Users/camilanunezrodriguez/Documents/GitHub/advanced_nlp_final/scripts/augmented_reviews.csv"  # part b (non-LLM augmentation)
TEXT_COL = "review"
LABEL_COL = "stars"
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
seed = 123


def load_extra(path, sep=",", text_col="review", label_col="stars"):
    df = pd.read_csv(path, sep=sep)
    df = df.rename(columns={text_col: TEXT_COL, label_col: LABEL_COL})
    df[TEXT_COL] = df[TEXT_COL].apply(preprocess_for_bert)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    return df[TEXT_COL].tolist(), (df[LABEL_COL].astype(int) - 1).tolist()


def load_splits():
    df = pd.read_csv(REAL_CSV)
    df[TEXT_COL] = df[TEXT_COL].apply(preprocess_for_bert)
    labeled = df.dropna(subset=[LABEL_COL, TEXT_COL]).reset_index(drop=True)
    texts = labeled[TEXT_COL].tolist()
    labels = (labeled[LABEL_COL].astype(int) - 1).tolist()

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=seed
    )

    limited_n = min(32, len(train_texts))
    try:
        small_texts, _, small_labels, _ = train_test_split(
            train_texts,
            train_labels,
            train_size=limited_n,
            stratify=train_labels,
            random_state=seed,
        )
    except Exception:
        small_texts, small_labels = train_texts[:limited_n], train_labels[:limited_n]

    synth_t, synth_l = load_extra(SYNTH_CSV)
    aug_t, aug_l = load_extra(
        AUG_CSV, sep="\t", text_col="text", label_col="star_rating"
    )
    return (
        small_texts,
        small_labels,
        synth_t,
        synth_l,
        aug_t,
        aug_l,
        valid_texts,
        valid_labels,
    )


def to_ds(texts, labels):
    return Dataset.from_dict({"text": list(texts), "label": list(labels)})


def train_setfit(train_texts, train_labels, valid_texts):
    # FULL SetFit (modern API >=1.1): contrastive body fine-tuning + head.
    # num_iterations now lives inside TrainingArguments; Trainer replaces
    # the old SetFitTrainer and takes args + column_mapping.
    model = SetFitModel.from_pretrained(ST_MODEL)
    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        num_iterations=20,  # contrastive pairs generated per example
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=to_ds(train_texts, train_labels),
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()
    return [int(p) for p in trainer.model.predict(valid_texts)]


def main():
    np.random.seed(seed)
    (real_t, real_l, synth_t, synth_l, aug_t, aug_l, valid_t, valid_l) = load_splits()
    print(
        f"Real:{len(real_t)} Aug(b):{len(aug_t)} Synth(d):{len(synth_t)} "
        f"Valid:{len(valid_t)}"
    )

    configs = {
        "real only": (real_t, real_l),
        "real + augmented(b)": (real_t + aug_t, real_l + aug_l),
        "real + synthetic(d)": (real_t + synth_t, real_l + synth_l),
        "real + both": (real_t + aug_t + synth_t, real_l + aug_l + synth_l),
    }

    metrics_val = Metrics()
    preds_by_config = {}
    for name, (tr_t, tr_l) in configs.items():
        print(f"\n=== Training SetFit: {name} ({len(tr_t)} examples) ===")
        preds = train_setfit(tr_t, tr_l, valid_t)
        preds_by_config[name] = preds
        metrics_val.run(valid_l, preds, name)

    print("\n--- Ordinal extras (MAE lower=better, QWK higher=better) ---")
    for name, preds in preds_by_config.items():
        mae = mean_absolute_error(valid_l, preds)
        qwk = cohen_kappa_score(valid_l, preds, weights="quadratic")
        print(f"  {name:22s}  MAE={mae:.4f}  QWK={qwk:.4f}")

    print("\n================  PART E — REAL SetFit  ================")
    print(pd.DataFrame(metrics_val.results).T.round(2))
    metrics_val.plot()


if __name__ == "__main__":
    main()


"""
Write up:


Based on what we saw form the results in part a to d, the main issue is overfitting due to too little data. 
So our idea for part e is to apply two target fixes together: a method built for tiny data, SetFit,  and to gradually incorporate more data.

We use the **setfit** package with **all-MiniLM-L6-v2**. To identify the most effective thechnique we compare four training configurations:

- using only the 32 labeled examples only
- using the 32 labeled examples plus the part b augmented data
- using the 32 labeled examples plus the part c synthetic data
- using the 32 labeled examples plus both the part b and c genereated data

This helps us isolate each augmentation source contribution on roder to tell which ones gives is the most added value. All four scenarios are 
trained with the same SetFit setup and evaluated in the same *7* real validation reviews

As we know, SetFit fine tunes a sentence-transforemer instead of a standard classification model. It trains in two stages, first it does 
the constrastive fine tuning of the encoder, which generates pairs of examples. For our case this means different rating pairs are pushed apart and similar
rating pairs are pulled together. The advantage of this is that it has a multiplicative effect of the learning signal since it turns the 25 exaples into hundreds
of training pairs. The second aprt is a classifier head is trained on the resulting embeddings.

This method is well suited for tiny data since the pairwise training extracts much more signal from few labels and 
we start from a pretrained sentence embedding which means less to learn from scratch, which will reduce the overfitting seen in parts a to d.






"""
