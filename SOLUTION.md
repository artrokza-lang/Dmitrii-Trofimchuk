# SOLUTION.md – SMILES-2026 Hallucination Detection

## Reproducibility instructions

1. **Open Google Colab** with GPU runtime (T4 recommended).  
2. **Run the following commands in a single cell** (or execute step by step):

```bash
!git clone https://github.com/artrokza-lang/Dmitrii-Trofimchuk.git
%cd Dmitrii-Trofimchuk
!pip install -r requirements.txt
!python solution.py
from google.colab import files; files.download('predictions.csv'); files.download('results.json')
```
## Final solution description
### Modified files
aggregation.py – feature extraction from hidden states.

probe.py – binary classifier (HallucinationProbe).

splitting.py – train/validation split strategy (5‑fold stratified cross‑validation).

## Final approach
### Feature extraction (aggregation.py)

Use last token of the final transformer layer (last real token, ignoring padding).

Additionally extract geometric features: for each layer compute the mean norm of the token vectors (over all real tokens), then calculate the standard deviation of these norms across layers and the difference between the last two layers.

Concatenate the last‑token vector (896‑dim) with the two geometric features → 898‑dimensional feature vector.

### Classifier (probe.py)

MLP architecture: Linear(898 → 512) → ReLU → Dropout(0.3) → Linear(512 → 256) → ReLU → Dropout(0.2) → Linear(256 → 1).

StandardScaler applied to all features.

Balanced BCE loss (positive weight = ratio of negative/positive samples).

Optimizer: Adam (lr = 0.001, weight decay = 1e-5).
Threshold tuning on the training set (search over 0.30–0.70, step 0.02) to maximise F1.

No early stopping (fixed 200 epochs) – simpler and more stable.

### Splitting (splitting.py)

5‑fold stratified cross‑validation. Each fold uses 4 folds for training, 1 fold for validation (the reported metrics are averaged).

The final test set (unlabelled) is predicted using a model trained on all 689 labelled samples.
## Why these choices work best
Last token of the final layer naturally captures the entire generated response (the model pools all information into the last position). It avoids the need to locate the start of the assistant’s answer, which is impossible with the available attention_mask (padding mask only).

Geometric features (std of layer norms, difference between last two layers) provide complementary information about how the internal representations evolve across layers, improving robustness.

Simple MLP with dropout regularises well on a small dataset (689 samples); deeper networks or BatchNorm caused overfitting.

5‑fold cross‑validation yields more stable and generalisable metrics than a single fixed split.

## Main contributor to metric improvement
Using the last token instead of mean pooling over all tokens (or over the last 50% tokens) gave the largest boost – from ~55% AUROC to 71.72% AUROC. Adding geometric features contributed an extra ~1% AUROC.

## Experiments and failed attempts
| Idea | Result | Why discarded |
|------|--------|----------------|
| Mean pooling over all tokens (last layer) | AUROC ~60% | Diluted signal from prompt and irrelevant tokens. |
| Mean pooling over last 50% tokens (heuristic) | AUROC ~68% | Inaccurate because response length varies. |
| Concatenated last‑token from last 4 layers | AUROC 68.0% | Increased dimensionality caused overfitting; no gain. |
| Deeper MLP (1024→512→256) with BatchNorm and LR scheduler | AUROC 68.4% | Overfitted; training became unstable. |
| Linear probe (logistic regression) | AUROC ~65% | Too simple; cannot capture non‑linear patterns. |
| k‑fold with simple MLP (without geometric features) | AUROC 71.0% | Slightly worse than 5‑fold + geometric features. |
| Variant D (mean pooling) – runtime error | – | Code incompatible with provided `attention_mask`. |
| Variant F (last token from layers 12,16,20,24) – runtime error | – | Index error; not pursued. |


## Final validation metrics (5‑fold average)
Test AUROC: 71.72%

Test Accuracy: 72.71%

Test F1: 80.52%

These numbers are reproducible with the provided code and random seed 42.

## Repository contents
aggregation.py, probe.py, splitting.py – student‑implemented files.

solution.py, model.py, evaluate.py – fixed infrastructure (not modified).

data/dataset.csv, data/test.csv – competition data.

predictions.csv – final submission (100 samples).

results.json – evaluation metrics (averaged over 5 folds).

requirements.txt – Python dependencies.







5‑fold cross‑validation yields more stable and generalisable metrics than a single fixed split.
