# Tactical Plan — 22DM015 Advanced Methods in NLP Final Project

**Course:** 22DM015 Advanced Methods in NLP, Barcelona School of Economics
**Instructor:** Arnault Gombert
**Deadline:** Defense on June 25 or 26, 2026
**Weight:** 70% of final grade
**Format:** Jupyter notebook(s) hosted on GitHub + 10-min presentation + 5-min Q&A

---

## 1. Plain-English Summary

This project asks you to tackle text classification on a HuggingFace dataset under a severe constraint: only 32 labeled examples. You progress through increasingly sophisticated techniques — from random and rule-based baselines, to fine-tuning BERT on 32 labels, to data augmentation, zero-shot LLM classification, and LLM-generated synthetic data — measuring how each one moves the needle. Then you unlock the full dataset, train at various percentages to draw learning curves, and compare everything. Finally, you compress your best model via distillation or quantization and analyze the speed-accuracy tradeoff. The deliverable is a well-documented GitHub repo with notebooks, an executive summary, and a short defense presentation.

---

## 2. Course Fit Mapping

| Criterion (points) | What you need to show | Where it appears |
|---|---|---|
| Part 1 — Setting Up (1.5 pts) | Bibliography/SOA, dataset description, random classifier, rule-based baseline | Notebook 1 |
| Part 2 — Data Scientist Challenge (3.5 pts) | BERT on 32 labels, augmentation, zero-shot LLM, LLM data generation, optimal technique | Notebook 2 |
| Part 3 — SOA Comparison (2 pts) | Full dataset training at 1%/10%/25%/50%/75%/100%, learning curve, technique comparison, methodology analysis | Notebook 3 |
| Part 4 — Distillation/Quantization (3 pts) | Compress best model, benchmark speed + performance, analysis + improvements | Notebook 4 |
| Presentation (implicit) | 10-min talk covering main findings, defend decisions | Slides |
| AI disclosure | Declare all AI usage | README or dedicated notebook section |
| Reproducibility | Random seed, clean notebook runs | All notebooks |

---

## 3. Progress Snapshot & Remaining Work

| Part | Sub-task | Status | Priority |
|---|---|---|---|
| **1a** Bibliography & SOA | ✅ Done | — |
| **1b** Dataset description | ✅ Done | — |
| **1c** Random classifier | ✅ Done | — |
| **1d** Rule-based baseline | ✅ Done | — |
| **2a** BERT on 32 labels | ✅ Done | — |
| **2b** Data augmentation (no LLM) | ✅ Done | — |
| **2c** Zero-shot LLM | ✅ Done | — |
| **2d** LLM data generation | ✅ Done | — |
| **2e** Optimal technique | ✅ Done | — |
| **3a** Full dataset training (incremental %) | ⚠️ Partial | Medium |
| **3b** Learning curve plot | ⚠️ Partial | Medium |
| **3c** Technique comparison on full data | ❌ Not started | Medium |
| **3d** Methodology analysis | ❌ Not started | **High** (1 pt, student-authored) |
| **4a** Distillation / quantization | ❌ Not started | **High** (1.5 pts) |
| **4b** Performance & speed comparison | ❌ Not started | **High** (0.5 pts) |
| **4c** Analysis & improvements | ❌ Not started | **High** (1 pt, student-authored) |
| **—** Executive summary | ❌ Not started | **High** |
| **—** Presentation slides | ❌ Not started | **High** |
| **—** AI disclosure | ❌ Not started | Medium |

**Remaining points at stake:** ~5 pts from Parts 3–4 + presentation readiness. This is the bulk of the grade weight you haven't locked in yet.

---

## 4. Deliverables Checklist

| Deliverable | Format | Location | Status |
|---|---|---|---|
| Notebook(s) with all code and markdown analysis | `.ipynb` | GitHub repo | In progress |
| Learning curve plot (Part 3b) | Figure in notebook | Notebook 3 | Not done |
| Technique comparison table/figure (Part 3c) | Table + figure in notebook | Notebook 3 | Not done |
| Written methodology analysis (Part 3d) | Markdown cells | Notebook 3 | Not done |
| Distilled/quantized model artifact | Saved model file | GitHub repo | Not done |
| Speed benchmark results (Part 4b) | Table in notebook | Notebook 4 | Not done |
| Written analysis of distillation (Part 4c) | Markdown cells | Notebook 4 | Not done |
| Executive summary | Markdown in README or notebook | GitHub repo top-level | Not done |
| Presentation slides | PDF/PPTX | GitHub repo or separate | Not done |
| AI disclosure statement | Markdown | README or notebook appendix | Not done |
| Random seed set at top of code | Code cell | All notebooks | Verify |

---

## 5. Project Structure & Notebook Organization

**Don't reorganize working code with 9 days left.** Instead, make two targeted moves:

1. **Separate Part 2 and Part 3 into distinct notebooks** if they're currently mixed. Cleanest approach: copy the notebook, delete Part 3 cells from one copy and Part 2 cells from the other. Each notebook loads data independently and saves its own results.

2. **Add a README table** so the grader knows exactly what to open and in what order. Having supporting scripts (`train.py`, `utils.py`) alongside notebooks is fine — just make the notebooks the narrative spine.

```text
nlp-final-project/
├── README.md                  # Executive summary + AI disclosure + structure table
├── 01_setup_and_baselines.ipynb       # Part 1
├── 02_limited_data_challenge.ipynb    # Part 2
├── 03_full_data_comparison.ipynb      # Part 3
├── 04_distillation_quantization.ipynb # Part 4
├── src/                       # Shared utilities (data loading, eval, training)
├── data/
│   ├── raw/
│   └── processed/
├── results/                   # Saved metrics JSON/CSV from each part
├── models/                    # Saved model checkpoints
├── figures/                   # Exported plots
├── presentation/              # Slides
└── requirements.txt
```

**README structure table (add this to your repo):**

```markdown
## Repository Structure

| File | Covers | Run order |
|------|--------|-----------|
| `01_setup_baselines.ipynb` | Part 1: SOA, dataset, random/rule-based baselines | 1 |
| `02_limited_data.ipynb` | Part 2: BERT-32, augmentation, zero-shot, LLM gen | 2 |
| `03_full_data_comparison.ipynb` | Part 3: learning curves, technique comparison | 3 |
| `04_distillation.ipynb` | Part 4: distillation/quantization | 4 |
| `src/utils.py` | Shared helpers (data loading, eval, cleanup) | — |
| `src/train.py` | Training loop (if applicable) | — |
```

**Key rule:** if the grader can open the README, see this table, and know exactly where everything lives, you're fine. Don't spend time on a bigger refactor.

---

## 6. Metric Strategy

**Primary metric:** F1-macro (or weighted F1) — standard for multi-class text classification, accounts for class imbalance without being dominated by the majority class.

**Also track:** Accuracy, per-class precision/recall, inference time (ms/sample), model size (MB/parameters).

**For Part 4 specifically:** the comparison is a 2×2 of (performance metric) × (efficiency metric). A good table shows: model name, F1, accuracy, parameters, size on disk, inference time per sample. This makes the distillation tradeoff immediately visible.

**Pitfall to avoid:** Don't only report accuracy if classes are imbalanced — the grader will notice.

---

## 7. Tactical Step-by-Step Plan (Remaining Work)

### Phase 1: Finish Part 3 (Target: 2 days)

This phase has four sub-tasks. Steps 1–8 cover the code/execution work (3a, 3b, 3c). Steps 9–10 cover the written analysis (3d) which must be student-authored.

---

#### Part 3a — Full Dataset Training (0.25 pts)

**Step 1: Set up the training infrastructure**
- Open (or create) `03_full_data_comparison.ipynb`.
- At the top: set random seed, import libraries, load your dataset and test split.
- Verify this is the **exact same test split** you used in Part 2. If you used a fixed seed + `train_test_split`, replicate that call. If you saved the split indices, load them.

**Step 2: Write a reusable training function**
- If you don't already have one, write a `train_and_evaluate(model, train_data, test_data)` function that returns a dict of metrics `{"f1": ..., "accuracy": ..., ...}`.
- This avoids copy-pasting the same training loop 6 times and reduces bugs.

**Step 3: Run the 6 incremental training runs**
- Define `percentages = [0.01, 0.10, 0.25, 0.50, 0.75, 1.0]`.
- For each percentage: sample that fraction of the training data (use `stratify` to preserve class distribution), train a fresh BERT model, evaluate, store results.
- **Critical:** delete the model and clear GPU memory between every run (see Section 13).
- Save the results dict to `results/part3a_learning_curve.json` after the loop so a kernel restart doesn't lose everything.

**Step 4: Verify the results make sense**
- Quick sanity check: performance should generally increase with more data. If 1% outperforms 50%, something is wrong with your sampling or evaluation.
- Add a markdown cell with a simple table showing the raw numbers before plotting.

---

#### Part 3b — Learning Curve Plot (0.25 pts)

**Step 5: Plot the learning curve**
- x-axis: training data percentage (or number of samples). y-axis: F1-macro.
- Plot the 6 data points from Step 3 as a line with markers.
- Add **horizontal reference lines** (dashed, different colors) for each Part 2 technique:
  - BERT on 32 labels (Part 2a)
  - Augmented BERT (Part 2b)
  - Zero-shot LLM (Part 2c)
  - LLM-generated data BERT (Part 2d)
  - Best combined approach (Part 2e)
- Add a legend identifying every line/reference.
- Title and axis labels. Save the figure to `figures/learning_curve.png`.

**Step 6: Add a markdown cell interpreting the plot**
- Note where the full-data curve crosses each Part 2 reference line — this tells you "how much labeled data equals the value of technique X."
- This interpretation must be in your own words, but the plot itself should make it visually obvious.

---

#### Part 3c — Technique Comparison (0.5 pts)

**Step 7: Build the comparison table**
- Collect all Part 2 results (load from saved JSONs or hardcode from your Part 2 notebook if needed).
- Collect all Part 3a results.
- Create a single pandas DataFrame with columns: `Method`, `Training Samples`, `F1`, `Accuracy`, and any other metrics you tracked.
- Rows should include:
  - Random classifier (Part 1c)
  - Rule-based baseline (Part 1d)
  - BERT 32 labels (Part 2a)
  - Augmented BERT (Part 2b)
  - Zero-shot LLM (Part 2c)
  - LLM-gen BERT (Part 2d)
  - Best combo (Part 2e)
  - BERT @ 1%, 10%, 25%, 50%, 75%, 100%
- Display the table in the notebook. Also consider a grouped bar chart for visual comparison.

**Step 8: Re-run Part 2 techniques on larger data (if applicable)**
- The assignment says "incorporate the techniques tested in Part 2 into your training schema for comparison." This likely means: try augmentation and/or LLM-generated data **on top of** larger data subsets (not just the 32-label regime).
- At minimum, pick 2–3 data percentages (e.g., 10%, 50%, 100%) and re-run with your best Part 2 augmentation technique applied. Add these to the comparison table.
- This shows whether augmentation still helps when you have more real data (usually the gap shrinks — that's a valid and interesting finding).

---

#### Part 3d — Methodology Analysis (1 pt, student-authored)

**Step 9: Write the analysis in markdown cells**
- This is worth a full point and must be your own writing. Cover these questions:
  - Which technique gave the biggest lift in the 32-label regime? Why do you think so?
  - At what data percentage does vanilla BERT match or surpass your best Part 2 approach?
  - Do augmentation techniques still help at larger data sizes, or do they plateau?
  - What are the limitations of each approach (e.g., LLM cost, augmentation noise, zero-shot inconsistency)?
  - If you had to deploy a solution with limited labels in production, which approach would you recommend and why?

**Step 10: Connect back to the SOA from Part 1**
- Compare your best full-data result to the benchmarks you found in Part 1a. Are you close to SOA? If not, what's the gap and what could close it?
- One paragraph is enough here.

---

**Output:** Completed Notebook 3 with all training runs, learning curve plot, comparison table, and written analysis.

**Checks before moving on:**
- [ ] All 6 training runs completed and results saved to disk.
- [ ] Learning curve plot is clear, labeled, and includes reference lines for all Part 2 techniques.
- [ ] Comparison table includes every method from Parts 1, 2, and 3 — all on the same test set.
- [ ] Part 2 techniques re-tested on larger data subsets for Part 3c.
- [ ] Methodology analysis markdown is written in your own words and discusses effectiveness + limitations.
- [ ] All runs use the same test split for fair comparison.
- [ ] Memory cleanup (`gc.collect()` + `empty_cache()`) runs between every training job.

**Good AI prompts for this phase:**
```text
Here is my learning curve data. What features should I comment on — inflection points,
diminishing returns, crossover with the LLM baseline? Don't write my analysis for me,
just tell me what to look for.
```

```text
I need to compare these methods on the same test set. Help me structure a clean
pandas DataFrame and a grouped bar chart. Don't interpret the results.
```

---

### Phase 2: Part 4 — Distillation / Quantization (Target: 3–4 days)

This is the highest-stakes remaining section (3 points). Your professor's Session 7.3 notebook uses **Pruna** for model compression — this is your primary reference and the approach the grader expects to see. The plan below uses Pruna-based compression (pruning + quantization) as the main path, with knowledge distillation as a complementary option that enriches your analysis.

**Key reference:** `Session_7_3_reduce_BERT_model.ipynb` — follow this notebook's patterns for Pruna API usage, the `measure_inference_metrics` helper, and the `codecarbon` integration.

---

#### Part 4a — Model Distillation / Quantization (1.5 pts)

##### Track A: Pruna-Based Compression (primary — matches professor's approach)

**Step 1: Set up the notebook and establish baseline**
- Open (or create) `04_distillation_quantization.ipynb`.
- Set random seed, import libraries, load dataset with the same test split as all other parts.
- Load your best-performing model from Part 3 (100% data). This is your **original/teacher**.
- Install Pruna: `pip install pruna`.
- Install codecarbon for carbon tracking: `pip install codecarbon`.
- Evaluate the original model and record baseline metrics (F1, precision, recall, inference speed, RAM, GPU memory, carbon footprint).

**Step 2: Set up the evaluation function**
- Adapt the professor's `measure_inference_metrics` function from Session 7.3. It tracks:
  - Inference speed (samples/sec)
  - CPU memory usage (MB) via `psutil`
  - GPU memory usage (MB) via `torch.cuda.memory_allocated`
  - Carbon footprint (kg CO2eq) via `codecarbon.EmissionsTracker`
  - F1, Precision, Recall (macro)
- This function will be reused for every compressed variant, ensuring consistent measurement.

**Step 3: Define Pruna compression strategies**
- Pruna uses `SmashConfig` + `smash()` to apply compression. Define a list of strategies to test:

  | Strategy | Pruna config | What it does | Requirements |
  |----------|-------------|--------------|--------------|
  | Unstructured pruning | `{"pruner": "torch_unstructured"}` | Zeros out individual weights by magnitude | None |
  | Dynamic quantization | `{"quantizer": "torch_dynamic"}` | Quantizes weights to INT8 at runtime (CPU) | None |
  | LLM INT8 quantization | `{"quantizer": "llm_int8"}` | Mixed-precision INT8 with outlier handling | None |
  | Structured pruning | `{"pruner": "torch_structured"}` | Removes entire neurons/channels | None |

- Code pattern for each strategy:
  ```python
  from pruna import SmashConfig, smash
  import copy

  smash_config = SmashConfig(batch_size=32, device=device)
  smash_config["quantizer"] = "torch_dynamic"  # or "pruner" = "torch_unstructured", etc.

  model_copy = copy.deepcopy(model)  # always work on a copy
  compressed_pruna_model = smash(model=model_copy, smash_config=smash_config)

  # Extract the underlying PyTorch model from PrunaModel wrapper
  if hasattr(compressed_pruna_model, 'model'):
      compressed_model = compressed_pruna_model.model
  else:
      compressed_model = compressed_pruna_model
  ```

**Step 4: Apply each strategy and measure**
- Loop through strategies, applying each to a fresh `deepcopy` of the original model.
- After each compression, run `measure_inference_metrics` and store results.
- **Critical:** delete the compressed model and clear GPU memory between strategies.
- Save all results to `results/part4_compression_results.json`.

**Step 5: Known pitfall — FP16 / half precision**
- The professor's notebook shows that `{"quantizer": "half"}` crashes BERT because Pruna's half-precision wrapper converts `input_ids` (which must be Long/Int) to FP16, causing an embedding layer error: `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.HalfTensor`.
- **Skip FP16 via Pruna**, or note the bug in a markdown cell if you want to show awareness. If you want FP16, do it manually: `model.half()` then cast `input_ids` back to Long before passing — but this adds complexity for marginal benefit.
- Demonstrating that you tried it, hit the bug, and understood why it fails is itself a good Part 4c discussion point.

##### Track B: Knowledge Distillation (complementary — strengthens analysis)

**Step 6: Choose a student architecture**
- The student must be meaningfully smaller. Common choices:
  - `bert-base` (110M) → `distilbert-base-uncased` (66M) or `prajjwal1/bert-small` (~29M)
- Add a markdown cell documenting: teacher name, teacher size, student name, student size, why you chose this student.

**Step 7: Generate soft labels from the teacher**
- Run the teacher on the full training set in eval mode, save raw logits.
- Save `teacher_logits` to disk (`torch.save()`) so a kernel restart doesn't lose them.
- **After saving:** delete the teacher and clear GPU memory.

**Step 8: Implement the distillation loss and train**
- Combined loss: `L = α * soft_loss + (1 - α) * hard_loss`
  - Soft loss: KL divergence on logits softened at temperature T
  - Hard loss: standard cross-entropy on true labels
- Starting values: `T = 3.0`, `α = 0.5`
- Key code:
  ```python
  import torch.nn.functional as F

  def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
      soft_loss = F.kl_div(
          F.log_softmax(student_logits / T, dim=-1),
          F.softmax(teacher_logits / T, dim=-1),
          reduction='batchmean'
      ) * (T ** 2)
      hard_loss = F.cross_entropy(student_logits, labels)
      return alpha * soft_loss + (1 - alpha) * hard_loss
  ```
- Train the student, log loss and validation F1 per epoch.
- Save to `models/student_distilled/`.

**Step 9: (Optional) Apply Pruna compression on top of the distilled student**
- Take the distilled student and apply Pruna quantization/pruning to it.
- This gives you a stacking analysis: distillation + quantization = maximum compression.

---

#### Part 4b — Performance & Speed Comparison (0.5 pts)

**Step 10: Evaluate all model variants on the same test set**
- Use the same `measure_inference_metrics` function for every variant.
- Model variants to compare (you'll have at least 3–4, potentially more):
  - Original fine-tuned BERT (baseline)
  - Pruned (unstructured) via Pruna
  - Quantized (dynamic INT8) via Pruna
  - Quantized (LLM INT8) via Pruna
  - Distilled student (if you did Track B)
  - Distilled + quantized (if you did Step 9)

**Step 11: Count parameters and measure model size**
- Parameters:
  ```python
  def count_params(model):
      return sum(p.numel() for p in model.parameters())
  ```
- Size on disk:
  ```python
  import os
  model.save_pretrained("tmp_model_dir")
  size_mb = sum(
      os.path.getsize(os.path.join("tmp_model_dir", f))
      for f in os.listdir("tmp_model_dir")
  ) / (1024 * 1024)
  ```

**Step 12: Build the comparison table**
- Create a pandas DataFrame with one row per model variant. Include **all** the professor's metrics:

  | Model | F1 | Precision | Recall | Speed (samples/sec) | RAM (MB) | GPU Mem (MB) | CO2 (kg) | Size (MB) |
  |-------|------|-----------|--------|---------------------|----------|--------------|----------|-----------|
  | Original BERT | ... | ... | ... | ... | ... | ... | ... | ... |
  | Pruned (unstructured) | ... | ... | ... | ... | ... | ... | ... | ... |
  | Quantized (dynamic) | ... | ... | ... | ... | ... | ... | ... | ... |
  | Quantized (LLM INT8) | ... | ... | ... | ... | ... | ... | ... | ... |
  | Distilled student | ... | ... | ... | ... | ... | ... | ... | ... |

- Create a visual comparison: bar chart or radar chart showing the tradeoffs.
- Save to `figures/compression_comparison.png`.

**Step 13: (Bonus) Carbon footprint analysis**
- Following the professor's approach, calculate per-request CO2eq and extrapolate to web scale:
  - CO2eq per request = total emissions / total test samples
  - Requests equivalent to one transatlantic flight (~480 kg CO2eq)
  - Daily emissions at 1M users × 10 requests/day
- This directly mirrors the professor's Session 7.3 analysis and shows you understood the sustainability angle. A few lines of calculation and a markdown cell is enough.

---

#### Part 4c — Analysis & Improvements (1 pt, student-authored)

**Step 14: Error analysis — where do compressed models fail?**
- Generate predictions from the original and each compressed variant on the test set.
- Compare confusion matrices side-by-side. Which classes degrade most under compression?
- For pruning specifically: check if the sparsity pattern affects certain linguistic features (e.g., shorter vs. longer texts, ambiguous vs. clear-cut examples).
- Add a markdown cell with the confusion matrices and your observations.

**Step 15: Efficiency analysis**
- For each compression strategy, calculate:
  - Compression ratio: `original_size / compressed_size`
  - Speedup ratio: `compressed_speed / original_speed` (in samples/sec)
  - Performance retention: `compressed_F1 / original_F1`
  - Carbon reduction: `1 - (compressed_CO2 / original_CO2)`
- Summarize: "Strategy X achieved Yх speedup with Z% F1 retention and W% carbon reduction."

**Step 16: Write the improvements section**
- This is the "suggest potential improvements or further research directions" part. Ideas to consider (pick 3–4 relevant to your results):
  - **Combine pruning + quantization:** Pruna supports setting both `pruner` and `quantizer` in the same SmashConfig — test the stacking effect.
  - **Structured vs. unstructured pruning:** unstructured pruning zeros individual weights (reduces effective parameters but not model size on disk); structured pruning removes entire neurons/heads (actually shrinks the architecture). Discuss the tradeoff.
  - **Quantization-aware training (QAT):** instead of post-training quantization, train with quantization in the loop for better accuracy retention.
  - **Intermediate layer distillation:** match hidden states, not just logits (TinyBERT-style).
  - **Progressive distillation:** distill in stages (large → medium → small) instead of one jump.
  - **Temperature tuning:** try different T values and report sensitivity.
  - **Pruning ratio tuning:** experiment with different sparsity levels (e.g., 30%, 50%, 70%) and plot the accuracy-sparsity tradeoff curve.
- For each suggestion, briefly explain what it is and why it might help given your specific results.

**Step 17: Write a concluding markdown cell**
- 3–5 sentences summarizing Part 4: what strategies you tested, which gave the best efficiency-performance tradeoff, the key numbers (e.g., "Dynamic quantization achieved 2× speedup with <1% F1 drop and 40% carbon reduction"), and your top improvement recommendation.

---

**Output:** Completed Notebook 4 with Pruna compression code, (optional) distillation code, comparison table, speed/memory/carbon benchmarks, and written analysis.

**Checks before moving on:**
- [ ] Original model evaluated as baseline with full metrics (F1, speed, memory, carbon).
- [ ] At least 2–3 Pruna compression strategies applied and evaluated.
- [ ] (Recommended) Knowledge distillation implemented as complementary approach.
- [ ] Known FP16 bug documented or avoided.
- [ ] Comparison table includes: model name, F1, precision, recall, speed, RAM, GPU memory, CO2, size.
- [ ] `measure_inference_metrics` function adapted from professor's notebook and used consistently.
- [ ] Carbon footprint per-request calculation included.
- [ ] Confusion matrices compared between original and compressed models.
- [ ] Written analysis covers deficiencies, efficiency tradeoff, and 3–4 improvement suggestions.
- [ ] All code has random seed and is reproducible.
- [ ] Memory cleanup between compression runs.

**Good AI prompts for this phase:**
```text
I'm using Pruna's SmashConfig to apply torch_dynamic quantization to my BERT model.
The compression runs but evaluation gives unexpected results. Here's my code. Help me
debug — don't write the full solution.
```

```text
I want to implement knowledge distillation for a BERT text classifier. Explain the
distillation loss function (KL divergence on softened logits + cross-entropy on hard labels)
and how temperature scaling works. Don't write the training loop for me — give me the
conceptual pieces I need to assemble.
```

```text
My Pruna half-precision quantization crashes with a RuntimeError about HalfTensor
in the embedding layer. Explain why this happens with BERT and what workarounds exist.
```

```text
Here are my original vs compressed model confusion matrices. What patterns should I
investigate for my analysis section? Don't write the analysis — just tell me what to look for.
```

---

### Phase 3: Executive Summary & AI Disclosure (Target: 0.5 days)

**Steps:**

1. Write a concise executive summary (half a page to one page): what dataset, what problem, what you tried, what worked best, what the distillation/quantization result was. This must be student-authored per the assignment rules.
2. Add an AI disclosure section listing every AI tool used, what it was used for, and where.
3. Put both in the README or a dedicated notebook section.

**Checks:**
- [ ] Executive summary covers objective, methods, key findings, and compression results.
- [ ] AI disclosure is complete and honest.

---

### Phase 4: Presentation Prep (Target: 1 day)

**Steps:**

1. Build 8–10 slides: problem setup (1), dataset (1), Part 2 results summary (2), learning curve + full-data comparison (1–2), distillation/quantization results (2), conclusions + future work (1).
2. Rehearse to fit in 10 minutes. Practice answering: "Why did you choose this augmentation technique?", "What was the biggest performance gap in the student model?", "If you had more time, what would you try?"
3. Anticipate Q&A questions about your methodology choices and be ready to defend them.

**Checks:**
- [ ] Slides tell a clear story: problem → methods → results → compression → takeaways.
- [ ] Every figure in the slides also exists in the notebooks.
- [ ] You can explain every number on every slide.
- [ ] Talk fits in 10 minutes.

---

## 8. Hypotheses

**H1:** Fine-tuning BERT on the full dataset will substantially outperform all 32-label approaches from Part 2.

**H2:** LLM-based techniques (zero-shot or data generation) will outperform non-LLM augmentation in the 32-label regime.

**H3:** Pruna-based compression (pruning + quantization) can retain F1 within 2–3 points of the original while reducing memory usage, inference time, and carbon footprint by at least 30–40%.

---

## 9. Possible Outcomes for Part 4

**Outcome A — Compression works well:** One or more Pruna strategies achieve near-original F1 with significant speed/memory/carbon gains. Dynamic quantization typically performs well here. Interpretation: post-training compression is viable for this task; deployment of a lighter model is practical.

**Outcome B — Partial success with tradeoffs:** Some strategies (e.g., aggressive pruning) lose noticeable performance while others (e.g., dynamic quantization) retain it. Interpretation: the compression landscape is strategy-dependent; recommend the best tradeoff point and suggest fine-tuning the compression parameters.

**Outcome C — FP16 crashes, others mixed:** The half-precision strategy fails due to the BERT embedding dtype bug (as in the professor's notebook), while INT8 methods work but with varying quality. Interpretation: not all compression methods are plug-and-play for encoder models; documenting failures is itself valuable.

**Outcome D — Distillation adds value beyond compression:** If you do both tracks, the distilled student outperforms a simply-compressed original at similar size. Interpretation: knowledge transfer captures more than weight compression alone.

---

## 10. Figure Strategy

**Centerpiece figure:** Learning curve (Part 3b) showing F1 vs. training data %, with horizontal reference lines for each Part 2 technique. This single figure tells the story of how much data you need and whether clever low-data techniques can substitute for more labels.

**Supporting figures:**
- Bar chart or table comparing all Part 2 methods side-by-side on the same test set.
- Part 4 compression comparison table/chart: original vs. pruned vs. quantized vs. distilled on F1, speed, memory, carbon.
- (Recommended) Per-request carbon footprint calculation and web-scale extrapolation (mirrors professor's Session 7.3 analysis).
- (Optional) Confusion matrices for original vs. best compressed model to show where compression hurts.
- (Optional) Class-level F1 breakdown across compression strategies.

---

## 11. Where AI Can Help

- Debugging code (training loops, data loading, quantization setup).
- Explaining concepts (distillation loss, temperature scaling, quantization types).
- Reviewing your draft analysis: "Is my interpretation consistent with these numbers?"
- Structuring comparison tables and figures.
- Suggesting what to look for in results.

## 12. Where AI Should Not Replace You

- **Part 3d methodology analysis** — must be your own written analysis.
- **Part 4c analysis and improvements** — must be your own interpretation.
- **Executive summary** — must be student-authored per submission guidelines.
- **Explaining your decisions in the defense** — you need to understand everything you submit.

---

## 13. Memory Management (Preventing Notebook Crashes)

Running multiple BERT training jobs in a single notebook will accumulate GPU/RAM usage and eventually OOM. This is the real crash risk — not notebook file size.

**Rule: clean up after every training run.**

```python
import gc, torch

del model, optimizer, trainer  # delete heavy objects
gc.collect()
torch.cuda.empty_cache()
```

**Part 2:** Add a cleanup cell between each experiment block (2a → 2b → 2c → 2d → 2e). If the notebook already runs top-to-bottom without crashing, leave it alone.

**Part 3 (higher risk):** Six training runs at increasing data sizes, plus Part 3c technique comparison runs. Use a loop pattern so each model is deleted before the next one loads:

```python
results = {}
for pct in [0.01, 0.10, 0.25, 0.50, 0.75, 1.0]:
    model = load_fresh_model()
    train_subset = sample_data(train_data, pct)
    metrics = train_and_evaluate(model, train_subset, test_data)
    results[pct] = metrics
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Save results so you don't re-run if kernel restarts
import json
with open("part3_results.json", "w") as f:
    json.dump(results, f)
```

Apply the same loop pattern for Part 3c when re-running with augmentation/LLM techniques at each data percentage.

**Part 4:** Less risky since you're only training one student model, but still clean up the teacher after extracting soft labels if you need the GPU for the student.

**General safeguards:**
- Save metrics and intermediate results to disk (JSON/CSV) after each experiment.
- Add markdown cells saying "restart kernel here if needed" between major sections.
- If using Colab, check GPU memory with `!nvidia-smi` between runs to catch leaks early.

---

## 14. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pruna FP16 crashes on BERT embedding layer | High (confirmed in professor's notebook) | Low | Skip FP16 via Pruna; document the bug in Part 4c as a known limitation |
| Pruna installation or version compatibility issues | Medium | Medium | Pin version from professor's environment; fallback to `torch.quantization.quantize_dynamic` |
| Distillation training is slow / crashes on available hardware | Medium | Medium | Use DistilBERT (small) as student; reduce batch size; use Colab GPU if local isn't enough |
| Compressed model performs much worse than original | Medium | Low | This is a valid finding — pre-write the Outcome B/C interpretation |
| codecarbon fails or gives zeros | Medium | Low | Carbon tracking is a bonus metric; proceed without it if needed, note in markdown |
| Running out of time before slides | High | High | Timebox Part 4 implementation to 3 days max; slides can reuse notebook figures directly |
| Inconsistent test splits across experiments | Low | High | Verify same test set is used everywhere; set random seed |

---

## 15. Timeline (June 16 → June 24)

| Day | Date | Focus |
|---|---|---|
| Day 1 | June 16 (Tue) | Split Part 2/3 into separate notebooks if mixed. Finish Part 3a–3b: complete remaining training runs, plot learning curve |
| Day 2 | June 17 (Wed) | Part 3c–3d: technique comparison table, write methodology analysis |
| Day 3 | June 18 (Thu) | Part 4a: set up Pruna, apply unstructured pruning + dynamic quantization, evaluate |
| Day 4 | June 19 (Fri) | Part 4a continued: LLM INT8, (optional) knowledge distillation, evaluate all variants |
| Day 5 | June 20 (Sat) | Part 4b: speed benchmarks, comparison table |
| Day 6 | June 21 (Sun) | Part 4c: write analysis and improvements. Draft executive summary |
| Day 7 | June 22 (Mon) | Build slides. Clean notebooks. Add AI disclosure |
| Day 8 | June 23 (Tue) | Rehearse presentation. Final notebook QA pass |
| Day 9 | June 24 (Wed) | Buffer day. Push final repo. Last rehearsal |
| Day 10 | June 25–26 | **Defense** |

---

## 16. Final QA Checklist

- [ ] Every sub-question (1a through 4c) is answered in the notebooks.
- [ ] All figures and tables are clearly labeled with titles and axis labels.
- [ ] Learning curve includes reference lines for Part 2 techniques.
- [ ] Distillation/quantization comparison table includes F1, size, speed.
- [ ] Written analysis in Parts 3d and 4c is in your own words.
- [ ] Executive summary is concise and covers objective, methods, findings, compression.
- [ ] AI disclosure is present and complete.
- [ ] Random seed is set at the top of every notebook.
- [ ] All notebooks run top-to-bottom without errors.
- [ ] Memory cleanup cells exist between training runs in Parts 2, 3, and 4.
- [ ] Intermediate results saved to disk (JSON/CSV) so kernel restarts don't lose progress.
- [ ] No hardcoded local paths.
- [ ] `requirements.txt` is up to date.
- [ ] GitHub repo is public (or shared with instructor).
- [ ] README includes a structure table mapping each file to its assignment part and run order.
- [ ] Parts 2 and 3 are in separate notebooks, each self-contained.
- [ ] Slides are ready and rehearsed (≤10 min).
- [ ] You can explain every method, metric, and result in the defense.

---

## 17. Suggested References

1. **Hinton et al. (2015)**, *Distilling the Knowledge in a Neural Network* — foundational distillation paper; cite for Part 4 methodology.
2. **Sanh et al. (2019)**, *DistilBERT, a distilled version of BERT* — directly relevant if using DistilBERT as student; cite for architecture choice.
3. **Devlin et al. (2019)**, *BERT: Pre-training of Deep Bidirectional Transformers* — cite for your base model throughout.
4. **Zafrir et al. (2019)**, *Q8BERT: Quantized 8Bit BERT* — relevant if pursuing quantization path.
5. **Pruna documentation**: https://docs.pruna.ai/en/stable/ — cite for compression tooling (SmashConfig, pruning, quantization strategies).
6. **codecarbon**: https://codecarbon.io/ — cite for carbon footprint measurement methodology.
7. **Dettmers et al. (2022)**, *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* — cite if using the `llm_int8` quantizer.
8. Whatever SOA paper you identified in Part 1a for your specific dataset/task.
