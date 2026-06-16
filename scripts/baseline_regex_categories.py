"""
Rule-based (regex/keyword) baseline for 22-class company-review category
classification.

Two separate "models" are built and evaluated:
  Model A: uses description           (company-level "about" text)
  Model B: uses description + title    (adds the customer's review headline)

Method: for each category, a set of weighted keyword/regex patterns is matched
against the (lowercased) text. Each match adds its weight to that category's
score. The predicted label is the argmax score; on a zero / tied score the
classifier falls back to the majority class.

No training is involved — the patterns are hand-crafted, so the full dataset is
used as the evaluation set.
"""

import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ---------------------------------------------------------------------------
# 1. Keyword / regex patterns per category
# ---------------------------------------------------------------------------
# Patterns are matched as whole-ish tokens via word boundaries. Multi-word
# phrases are allowed. Weights let strong signals (e.g. "vet" for pets) count
# more than weak/ambiguous ones. Keep terms lowercase; text is lowercased
# before matching.

CATEGORY_PATTERNS = {
    "Animals & Pets": [
        (
            r"\b(dog|cat|puppy|kitten|pet|pets|vet|veterinary|aquarium|kennel|"
            r"grooming|leash|collar|paw|breed|petfood|cattery)\b",
            1.0,
        ),
    ],
    "Beauty & Well-being": [
        (
            r"\b(salon|spa|skincare|skin|makeup|cosmetic|facial|hair(cut|dresser|"
            r"stylist)?|nails?|manicure|massage|wellbeing|wellness|fragrance|"
            r"serum|moisturiser|moisturizer)\b",
            1.0,
        ),
    ],
    "Business Services": [
        (
            r"\b(b2b|invoice|accounting|consultanc?y|consultant|payroll|"
            r"outsourc|procurement|saas|crm|logistics|wholesale|supplier|"
            r"corporate)\b",
            1.0,
        ),
    ],
    "Construction & Manufacturing": [
        (
            r"\b(construction|builder|building|contractor|manufactur|factory|"
            r"concrete|scaffold|roofing|brick|cement|joinery|fabricat|"
            r"machinery)\b",
            1.0,
        ),
    ],
    "Education & Training": [
        (
            r"\b(course|courses|training|tutor|tutoring|学|lesson|teacher|"
            r"teaching|school|college|university|exam|certification|diploma|"
            r"curriculum|e-learning|webinar|bootcamp|student)\b",
            1.0,
        ),
    ],
    "Electronics & Technology": [
        (
            r"\b(laptop|phone|smartphone|computer|pc|gadget|electronics?|tech|"
            r"software|hardware|router|monitor|charger|headphones?|battery|"
            r"device|app|gpu|usb)\b",
            1.0,
        ),
    ],
    "Events & Entertainment": [
        (
            r"\b(ticket|tickets|concert|festival|event|gig|show|venue|theatre|"
            r"theater|cinema|party|booking|performance|nightclub)\b",
            1.0,
        ),
    ],
    "Food, Beverages & Tobacco": [
        (
            r"\b(food|snack|coffee|tea|wine|beer|drink|beverage|chocolate|"
            r"grocery|groceries|cigarette|tobacco|vape|meal kit|delicatessen|"
            r"bakery|brewery)\b",
            1.0,
        ),
    ],
    "Health & Medical": [
        (
            r"\b(doctor|clinic|hospital|medical|medicine|pharmacy|prescription|"
            r"dentist|dental|gp|nhs|surgery|therapy|physio|patient|health|"
            r"diagnosis|treatment)\b",
            1.0,
        ),
    ],
    "Hobbies & Crafts": [
        (
            r"\b(craft|crafts|knitting|sewing|hobby|model kit|paint(ing)?|"
            r"scrapbook|yarn|fabric|beads?|pottery|woodworking|collectib)\b",
            1.0,
        ),
    ],
    "Home & Garden": [
        (
            r"\b(furniture|sofa|garden|gardening|plant|patio|kitchen|mattress|"
            r"bedding|curtain|décor|decor|lawn|shed|home decor|interior|"
            r"appliance)\b",
            1.0,
        ),
    ],
    "Home Services": [
        (
            r"\b(plumber|plumbing|electrician|cleaner|cleaning|removal|handyman|"
            r"pest control|locksmith|installation|repair man|gas engineer|"
            r"boiler)\b",
            1.0,
        ),
    ],
    "Legal Services & Government": [
        (
            r"\b(lawyer|solicitor|legal|attorney|court|conveyanc|will writing|"
            r"government|council|visa|passport|notary|litigation|tribunal|"
            r"barrister)\b",
            1.0,
        ),
    ],
    "Media & Publishing": [
        (
            r"\b(magazine|newspaper|publisher|publishing|subscription|book|"
            r"ebook|news|journal|press|streaming|podcast|author|editorial)\b",
            1.0,
        ),
    ],
    "Money & Insurance": [
        (
            r"\b(insurance|bank|banking|loan|mortgage|credit|investment|policy|"
            r"premium|claim|broker|pension|finance|financial|payment|refund|"
            r"savings|interest rate)\b",
            1.0,
        ),
    ],
    "Public & Local Services": [
        (
            r"\b(council|public service|library|recycling|waste|water board|"
            r"local authority|municipal|community service|registry)\b",
            1.0,
        ),
    ],
    "Restaurants & Bars": [
        (
            r"\b(restaurant|bar|pub|cafe|café|menu|waiter|waitress|dining|diner|"
            r"bistro|reservation|table|takeaway|takeout|chef|cocktail)\b",
            1.0,
        ),
    ],
    "Shopping & Fashion": [
        (
            r"\b(dress|shirt|shoes?|clothing|clothes|fashion|wardrobe|jeans|"
            r"jacket|boutique|apparel|size|fit|garment|outfit|accessor)\b",
            1.0,
        ),
    ],
    "Sports": [
        (
            r"\b(gym|fitness|football|soccer|tennis|cycling|bike|running|"
            r"workout|sport|sports|athletic|trainers?|golf|swimming|yoga|"
            r"equipment)\b",
            1.0,
        ),
    ],
    "Travel & Vacation": [
        (
            r"\b(hotel|flight|airline|booking|holiday|vacation|travel|trip|"
            r"resort|airbnb|cruise|tour|airport|luggage|itinerary|check-?in)\b",
            1.0,
        ),
    ],
    "Utilities": [
        (
            r"\b(electricity|energy|gas supplier|broadband|internet provider|"
            r"utility|utilities|water bill|tariff|meter|wifi|phone provider|"
            r"mobile network)\b",
            1.0,
        ),
    ],
    "Vehicles & Transportation": [
        (
            r"\b(car|cars|vehicle|garage|mechanic|tyre|tire|mot|dealership|"
            r"taxi|uber|bus|train|delivery driver|van|motorbike|automotive|"
            r"servicing)\b",
            1.0,
        ),
    ],
}

# Pre-compile for speed.
COMPILED = {
    cat: [(re.compile(pat), w) for pat, w in pats]
    for cat, pats in CATEGORY_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# 2. Classifier
# ---------------------------------------------------------------------------
def score_text(text):
    """Return {category: score} for one piece of text."""
    text = text.lower()
    scores = {}
    for cat, pats in COMPILED.items():
        s = 0.0
        for rx, w in pats:
            if rx.search(text):
                s += w
        if s:
            scores[cat] = s
    return scores


def predict(texts, fallback):
    """Vectorized-ish predict over an iterable of strings."""
    preds = []
    for t in texts:
        scores = score_text(t)
        if not scores:
            preds.append(fallback)
            continue
        best = max(scores.values())
        winners = [c for c, v in scores.items() if v == best]
        # On a tie, fall back to majority class for determinism.
        preds.append(winners[0] if len(winners) == 1 else fallback)
    return preds


# ---------------------------------------------------------------------------
# 3. Run + evaluate
# ---------------------------------------------------------------------------
def build_text(df, fields):
    return df[fields].fillna("").astype(str).agg(" ".join, axis=1)


def evaluate(df, fields, name):
    fallback = df["category"].value_counts().idxmax()
    text = build_text(df, fields)
    y_true = df["category"].tolist()
    y_pred = predict(text.tolist(), fallback)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'=' * 70}\n{name}\nfields = {fields}\n{'=' * 70}")
    print(f"accuracy    : {acc:.4f}")
    print(f"macro F1    : {macro_f1:.4f}")
    print(f"weighted F1 : {weighted_f1:.4f}")
    print(
        f"(random baseline ~ {1 / df['category'].nunique():.4f}, "
        f"majority-only ~ "
        f"{df['category'].value_counts(normalize=True).iloc[0]:.4f})"
    )
    print("\nper-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


if __name__ == "__main__":
    # Load your dataframe here, e.g.:
    #   df = pd.read_parquet("reviews.parquet")
    # or from the HF datasets library:
    #   from datasets import load_dataset
    #   df = load_dataset("<name>", split="train").to_pandas()
    df = pd.read_pickle(
        "/Users/camilanunezrodriguez/Documents/GitHub/advanced_nlp_final/data/trustpilot_reviews.pkl"
    )

    results = []
    # Model A: company description only
    results.append(evaluate(df, ["description"], "Model A (description)"))
    # Model B: description + review title
    results.append(
        evaluate(df, ["description", "title"], "Model B (description + title)")
    )

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(pd.DataFrame(results).to_string(index=False))
