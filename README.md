# TRUST-4127 · Comment-Guard v0 — the YouTube spam gate

A from-scratch (NumPy-only) perceptron spam gate for YouTube comments, plus an honest error
analysis and a costed ship/no-ship recommendation. Built for the Google Zürich · YouTube
Trust & Safety intern project (ticket GOOGL-204).

## What this is

Every minute thousands of comments hit YouTube. Most are fine; some are scam-bots; the nastiest
are **hijacked veteran accounts** — years-old, legit-looking accounts taken over to push scams.
This notebook builds the simplest model that works, shows exactly where it breaks, and quantifies
what it takes to fix it.

The four acts:
1. **EDA** — class balance (~25% spam) + feature-vs-label views.
2. **The gate** — a from-scratch NumPy perceptron (`fit`/`predict`, perceptron rule) on
   train-only-standardized features, reported with precision & recall, not just accuracy.
3. **The break** — slice the false negatives, find/name/plot the **hijacked-veteran** segment
   (old accounts, few links, only detectable via `dup_ratio × posts_last_hour`).
4. **The fix + the call** — engineer the interaction feature and/or train the tiny hidden-layer
   net, show the lift overall *and* on the veteran segment, and write a numbers-first recommendation.

## Files

- `zurich-f1-comment-guard-project.ipynb` — the completed notebook, runs top-to-bottom with all
  outputs (plots, tables, metrics) visible.
- `zurich-comment-guard.csv` — 2,400 comments, 8 engineered features + `is_spam` label.

## How to run

```bash
pip install numpy pandas matplotlib
jupyter notebook zurich-f1-comment-guard-project.ipynb   # Kernel → Restart & Run All
```

Or open it in Google Colab and Run All. Everything is deterministic (`seed=0`); no GPU needed.

## Headline numbers (test set, 25% holdout)

| model | accuracy | precision | recall | veteran-segment recall |
|---|---|---|---|---|
| majority baseline | 0.767 | 0.000 | 0.000 | 0.000 |
| perceptron | 0.902 | 0.758 | 0.850 | 0.541 |
| perceptron + cross | 0.903 | 0.753 | 0.871 | 0.514 |
| tiny net | 0.957 | 0.945 | 0.864 | 0.514 |
| **tiny net + cross** | **0.972** | **0.956** | **0.921** | **0.703** |

## The recommendation, in one line

**Ship the tiny hidden-layer net for v0** — it lifts precision from 0.76 → 0.95 (few legit
comments wrongly removed) at 0.96 accuracy. It still under-catches **hijacked veterans**
(old, link-free accounts; recall ~0.51), so route that suspect band to human review. For **v1**,
feed the net the `dup_ratio × posts_last_hour` interaction: in this run that single change raised
veteran recall to **0.70** with no loss elsewhere.

## Notes on method

- **No leakage:** the standardizer is fit on the training split only, then applied to both splits.
- **Perceptron honesty:** the data is not linearly separable, so the perceptron rule never drives
  its error count to zero (it oscillates) — which is precisely why the veteran pocket, visible only
  in a feature *interaction*, slips through a single linear layer.
- **Model code is from-scratch NumPy** (perceptron and the 1-hidden-layer net with hand-written
  backprop); no scikit-learn / PyTorch / TensorFlow is used for the models.
