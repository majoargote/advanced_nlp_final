"""
Part d — Data Generation with LLM (1 point)
============================================
Use an LLM (OpenAI) to generate new labeled examples, train the SAME BERT model
as part a on them + the 32 real labels, and analyze how this impacts metrics.

Pipeline (two stages; only Stage 1 touches the API):
  Stage 1  GENERATE  -> call OpenAI to synthesize reviews per (star, category)
           FILTER     -> drop generations the real-only judge mislabels badly
  Stage 2  TRAIN      -> fine-tune bert-base-uncased on real / synth / real+synth

Consistency with part a (this is what makes the comparison valid):
  - same checkpoint: bert-base-uncased ; same max_length: 64
  - same preprocess_for_bert applied to ALL text (real AND synthetic)
  - same seed (123) and same stratified 0.2 validation split -> same test set
  - same training args (20 epochs, lr 5e-5, eval each epoch, best-on-eval_loss)
  - same Metrics class from utils for the headline numbers
  - ADDED: ordinal-aware MAE and QWK (stars are ordered)

Data facts (from inspection of filtered_reviews.csv):
  - 32 labeled rows total (the only labeled data that exists)
  - 10,744 rows with review text but NO star label
  - two categories: "Travel & Vacation" and "Media & Publishing"

Seeding strategy (no leakage):
  - LABEL seeds: the 25 TRAINING rows, per (star, category). These teach the
    LLM what each star's text looks like. Validation rows are NEVER used.
  - STYLE seeds: a few UNLABELED reviews per category (from the 10,744). They
    have no labels, so they cannot leak the test labels; they only supply
    authentic topic/vocabulary for each category.

Setup:
  pip install openai python-dotenv nest_asyncio
  Put OPENAI_API_KEY in a .env file (NOT hardcoded, NOT committed).
"""

import os
import re
import json
import asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# allow asyncio.run() inside an interactive kernel (Jupyter / IPython / IDE cells)
try:
    import nest_asyncio

    nest_asyncio.apply()
except Exception:
    pass

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset as HFDataset

# same helpers part a uses
from utils import Metrics, preprocess_for_bert

# ----------------------------------------------------------------------------
# CONFIG  — matched to part a
# ----------------------------------------------------------------------------
REAL_CSV = "../data/filtered_reviews.csv"
TEXT_COL = "review"
LABEL_COL = "stars"
CAT_COL = "category"

checkpoint = "bert-base-uncased"  # SAME as part a
max_length = 64  # SAME as part a
num_labels = 5  # stars 1..5 -> labels 0..4

GEN_MODEL = "gpt-4o-mini"
PER_STAR = 500  # synthetic reviews per star (split across cats)
BATCH_SIZE_GEN = 10  # reviews per API call
MAX_CONCURRENT = 20
N_LABEL_SEEDS = 3  # real labeled examples shown per prompt
N_STYLE_SEEDS = 3  # unlabeled style examples shown per prompt

seed = 123  # SAME as part a


def set_seed(s=123):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)


set_seed(seed)
load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def star_to_label(star: int) -> int:
    return int(star) - 1


def label_to_star(label: int) -> int:
    return int(label) + 1


# ============================================================================
# STAGE 1a — GENERATE synthetic reviews per (star, category)
# ============================================================================
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

# Contrastive per-rating guidance (DINO idea). Describing what each star feels
# like — and how it differs from neighbours — is what makes a 4-star separable
# from a 3- or 5-star. The middle ratings are the hard part.
STAR_GUIDANCE = {
    1: (
        "a furious, deeply dissatisfied customer. Major problems, you feel "
        "cheated, no redeeming qualities. Strong negative language."
    ),
    2: (
        "a dissatisfied customer. Poor experience with significant problems, "
        "but one or two minor things were tolerable. Mostly negative, not as "
        "scathing as the very worst reviews."
    ),
    3: (
        "a neutral, mixed customer. Genuinely torn: some things fine, others "
        "disappointing. Balanced, lukewarm, neither recommending nor warning "
        "off. The most ambivalent tone."
    ),
    4: (
        "a satisfied customer with a minor caveat. Mostly positive, you'd "
        "recommend it, but note one small flaw. More positive than mixed, not "
        "gushing."
    ),
    5: (
        "a delighted customer. Everything exceeded expectations. Warm, glowing "
        "praise, would strongly recommend. No complaints."
    ),
}


def build_prompt(star, category, n, label_seeds, style_seeds):
    """label_seeds: real reviews of THIS star+category (may be few/empty).
    style_seeds: unlabeled reviews of THIS category (topic/vocab anchor)."""
    label_block = ""
    if label_seeds:
        joined = "\n".join(f'  - "{s}"' for s in label_seeds)
        label_block = (
            f"\nReal examples of {star}-star '{category}' reviews "
            f"(match this rating's tone):\n{joined}\n"
        )
    style_block = ""
    if style_seeds:
        joined = "\n".join(f'  - "{s}"' for s in style_seeds)
        style_block = (
            f"\nExamples of the topics/vocabulary real '{category}' "
            f"reviews use (any rating — copy the SUBJECT MATTER, not "
            f"the sentiment):\n{joined}\n"
        )

    return f"""You are generating realistic Trustpilot-style customer reviews in the category "{category}".

Write {n} DISTINCT reviews from the point of view of {STAR_GUIDANCE[star]}
{label_block}{style_block}
Requirements:
- Each review reflects EXACTLY a {star}-out-of-5 star rating in tone and content.
- Stay on-topic for the "{category}" category (same kind of companies/services as the examples).
- Vary length (1-5 sentences), wording, and what is praised or criticised. Avoid repeating openings.
- Sound like a real customer, not a critic. Do NOT mention the star number in the text.
- Output ONLY valid JSON: a list of objects, each {{"review": "<text>"}}. No markdown, no commentary."""


async def generate_for_cell(star, category, total, seeds, semaphore):
    """Generate `total` reviews for one (star, category) cell."""
    label_seeds = seeds["label"].get((star, category), [])
    style_seeds = seeds["style"].get(category, [])
    out = []

    async def one_batch(n):
        async with semaphore:
            # resample seeds each batch for variety
            ls = (
                list(
                    np.random.choice(
                        label_seeds, min(N_LABEL_SEEDS, len(label_seeds)), replace=False
                    )
                )
                if label_seeds
                else []
            )
            ss = (
                list(
                    np.random.choice(
                        style_seeds, min(N_STYLE_SEEDS, len(style_seeds)), replace=False
                    )
                )
                if style_seeds
                else []
            )
            prompt = build_prompt(star, category, n, ls, ss)
            try:
                resp = await async_client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=1500,
                )
                content = resp.choices[0].message.content.strip()
                content = re.sub(r"^```(json)?|```$", "", content).strip()
                items = json.loads(content)
                return [
                    {
                        "review": it["review"].strip(),
                        "stars": star,
                        "category": category,
                    }
                    for it in items
                    if isinstance(it, dict) and it.get("review", "").strip()
                ]
            except Exception as e:
                print(f"  [star {star} | {category}] batch error: {e}")
                await asyncio.sleep(2)
                return []

    n_batches = total // BATCH_SIZE_GEN + (1 if total % BATCH_SIZE_GEN else 0)
    tasks = [
        one_batch(min(BATCH_SIZE_GEN, total - i * BATCH_SIZE_GEN))
        for i in range(n_batches)
    ]
    for batch in await asyncio.gather(*tasks):
        out.extend(batch)
    return out[:total]


async def generate_all(categories, seeds):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    per_cat = PER_STAR // len(categories)  # split each star across categories
    rows = []
    for star in range(1, 6):
        for category in categories:
            print(f"Generating {per_cat} reviews | star {star} | {category}")
            rows.extend(await generate_for_cell(star, category, per_cat, seeds, sem))
    df = pd.DataFrame(rows)
    print(f"\nGenerated {len(df)} raw synthetic reviews")
    print(df.groupby(["stars", "category"]).size())
    return df


# ============================================================================
# Shared training helpers — mirror part a exactly
# ============================================================================
def to_hf_dataset(texts, labels=None):
    data = {"review": list(texts)}
    if labels is not None:
        data["labels"] = list(labels)
    return HFDataset.from_dict(data)


def tokenize(batch):
    return tokenizer(batch["review"], truncation=True, max_length=max_length)


def train_bert(
    train_texts,
    train_labels,
    valid_texts,
    valid_labels,
    num_train_epochs=20,
    output_dir="./tmp_bert_partd",
):
    train_ds = to_hf_dataset(train_texts, train_labels).map(tokenize, batched=True)
    valid_ds = to_hf_dataset(valid_texts, valid_labels).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        logging_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    preds = trainer.predict(valid_ds)
    valid_preds = preds.predictions.argmax(-1)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return valid_preds


def ordinal_extras(y_true_labels, y_pred_labels):
    return {
        "MAE": mean_absolute_error(y_true_labels, y_pred_labels),
        "QWK": cohen_kappa_score(y_true_labels, y_pred_labels, weights="quadratic"),
    }


# ============================================================================
# STAGE 1b — CONSISTENCY FILTER
# ============================================================================
def consistency_filter(synth_df, judge_texts, judge_labels, drop_distance=2):
    """Drop synthetic rows whose claimed star is >= drop_distance from a
    real-only judge's prediction. Judge is weak (trained on ~25 rows) so this
    removes only gross mismatches, not a clean ground truth."""
    judge_ds = to_hf_dataset(judge_texts, judge_labels).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )
    judge = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_judge",
            num_train_epochs=20,
            per_device_train_batch_size=16,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            logging_strategy="epoch",
        ),
        train_dataset=judge_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    judge.train()

    synth_ds = to_hf_dataset(synth_df["review"].tolist()).map(tokenize, batched=True)
    pred_labels = judge.predict(synth_ds).predictions.argmax(-1)
    pred_stars = [label_to_star(p) for p in pred_labels]

    del judge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    synth_df = synth_df.copy()
    synth_df["pred_star"] = pred_stars
    kept = synth_df[(synth_df["pred_star"] - synth_df["stars"]).abs() < drop_distance]

    print(
        f"\nConsistency filter: kept {len(kept)}/{len(synth_df)} "
        f"({100 * len(kept) / len(synth_df):.0f}%)"
    )
    print("Kept per star:")
    print(kept.groupby(["stars", "category"]).size())
    return kept.drop(columns=["pred_star"]).reset_index(drop=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    set_seed(seed)

    # --- load -----------------------------------------------------------------
    df = pd.read_csv(REAL_CSV)
    df[TEXT_COL] = df[TEXT_COL].apply(preprocess_for_bert)  # SAME preprocessing

    labeled = df.dropna(subset=[LABEL_COL, TEXT_COL]).reset_index(drop=True)
    unlabeled = df[df[LABEL_COL].isna() & df[TEXT_COL].notna()]
    categories = sorted(labeled[CAT_COL].dropna().unique().tolist())
    print(
        f"Labeled: {len(labeled)}   Unlabeled: {len(unlabeled)}   Categories: {categories}"
    )

    texts = labeled[TEXT_COL].tolist()
    labels = (labeled[LABEL_COL].astype(int) - 1).tolist()  # 0..4
    cats = labeled[CAT_COL].tolist()

    # Split the SAME way part a does (on the text/label lists, same seed), so
    # the validation set is IDENTICAL to part a's.
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=seed
    )

    # part a then caps the training set at 32 (stratified), with a fallback —
    # reproduce it exactly (the bare except mirrors part a when the split fails)
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

    # recover category + star for each chosen training row (needed for seeding)
    text_to_meta = {t: (c, l + 1) for t, c, l in zip(texts, cats, labels)}
    small_cats = [text_to_meta[t][0] for t in small_texts]
    small_stars = [text_to_meta[t][1] for t in small_texts]
    print(f"Real train: {len(small_texts)}   Validation: {len(valid_texts)}")
    if len(set(small_texts)) != len(small_texts):
        print(
            "WARNING: duplicate training texts; (cat, star) recovery may be ambiguous"
        )

    # --- build seed pools (no leakage) ---------------------------------------
    # LABEL seeds: real TRAINING reviews keyed by (star, category)
    label_seeds = {}
    for t, s, c in zip(small_texts, small_stars, small_cats):
        label_seeds.setdefault((s, c), []).append(t)
    # STYLE seeds: UNLABELED reviews keyed by category (no labels -> no leakage)
    style_seeds = {}
    for c in categories:
        pool = unlabeled[unlabeled[CAT_COL] == c][TEXT_COL].tolist()
        style_seeds[c] = (
            list(np.random.choice(pool, min(50, len(pool)), replace=False))
            if pool
            else []
        )
    seeds = {"label": label_seeds, "style": style_seeds}

    metrics_val = Metrics()

    # --- STAGE 1: generate + filter ------------------------------------------
    synth = asyncio.run(generate_all(categories, seeds))
    synth["review"] = synth["review"].apply(preprocess_for_bert)  # SAME preprocessing
    synth.to_csv("synthetic_raw.csv", index=False)

    synth = consistency_filter(synth, small_texts, small_labels)
    synth.to_csv("synthetic_filtered.csv", index=False)
    synth_texts = synth["review"].tolist()
    synth_labels = [star_to_label(s) for s in synth["stars"]]

    # --- STAGE 2: three configs, all evaluated on the SAME real valid set ----
    extras = {}

    p = train_bert(small_texts, small_labels, valid_texts, valid_labels)
    metrics_val.run(valid_labels, p, "BERT real-only (32)")
    extras["real_only"] = ordinal_extras(valid_labels, p)

    p = train_bert(synth_texts, synth_labels, valid_texts, valid_labels)
    metrics_val.run(valid_labels, p, "BERT synthetic-only")
    extras["synthetic_only"] = ordinal_extras(valid_labels, p)

    p = train_bert(
        small_texts + synth_texts,
        small_labels + synth_labels,
        valid_texts,
        valid_labels,
    )
    metrics_val.run(valid_labels, p, "BERT real+synthetic")
    extras["real_plus_synth"] = ordinal_extras(valid_labels, p)

    print("\n--- Ordinal extras (MAE lower=better, QWK higher=better) ---")
    print(pd.DataFrame(extras).T.round(4))

    metrics_val.plot()


if __name__ == "__main__":
    main()
