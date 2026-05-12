# SOLUTION.md – SMILES-2026 Hallucination Detection

## Reproducibility instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
   cd SMILES-HALLUCINATION-DETECTION
2. **Set up environment (Python 3.10+ recommended)**
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows
pip install -r requirements.txt
```
3. **Run the solution**
```bash
python solution.py
```
This will:

Load data/dataset.csv and data/test.csv

Extract hidden states from Qwen/Qwen2.5-0.5B

Train HallucinationProbe (MLP classifier)

Generate predictions.csv (same format as required)

Save results.json
4. **Expected output**
predictions.csv – submission file with id,label columns.

results.json – evaluation metrics on train/val/test.

Note: Running on a free Google Colab T4 GPU takes ~15 minutes. CPU execution is not recommended.
# Final solution description
## Components modified
aggregation.py – feature extraction from hidden states.

probe.py – binary classifier (HallucinationProbe).

splitting.py – train/validation/test split.

## Final approach
1. **Feature extraction (aggregation.py)**

Use only the last 4 transformer layers of Qwen2.5-0.5B (hidden size = 896).

Apply mean pooling exclusively over the assistant’s response tokens (mask out system and user parts).

Concatenate the 4 pooled vectors → a single 3584‑dimensional feature vector per sample.

No hand‑crafted geometric features were added (they did not improve validation F1).
2. **Classifier (probe.py)**
Architecture:
```bash
Linear(3584 → 512) → ReLU → Dropout(0.3)
→ Linear(512 → 256) → ReLU → Dropout(0.2)  
→ Linear(256 → 1)
```
StandardScaler applied to all features before training.

Class‑weighted BCE loss (pos_weight = ratio of negative/positive samples).

Early stopping (patience = 5) on validation loss.

Threshold tuning on validation set: search over 0.30–0.70 (step 0.02) to maximise F1 score.

Optimizer: Adam (lr = 0.001, weight decay = 1e-5).
3. **Splitting (splitting.py)**
Stratified train/validation/test split (80% / 15% / 15%) with fixed random seed.
## Why these choices worked best
Last 4 layers capture rich semantic and factual information (early layers are more syntactic, later layers more task‑oriented).

Mean pooling on response tokens focuses the probe on generated text only, ignoring the prompt.

Dropout + early stopping effectively prevented overfitting on the small dataset (689 samples).
Threshold tuning corrected the natural bias of the model towards the majority class.

## Main contributor to metric improvement
The combination of using the last 4 layers (instead of a single layer) and strict masking of response tokens gave the largest boost (≈ +5% F1). Standardisation and dropout also helped stabilise training.
# Experiments and failed attempts
## Attempted but discarded
1. **Geometric features**

Added token‑wise norms, standard deviations, response/prompt length ratios.

Result: No improvement (F1 stayed ~0.77), but increased feature dimension and training time.

Reason: The mean‑pooled hidden states already encode the necessary information; hand‑crafted statistics were redundant.

2. **Single‑layer probes**

Tried using only the last layer or the 12th layer.

Result: Validation F1 dropped to 0.72–0.74.

Reason: Different layers encode complementary information; concatenating them gives a more holistic representation.

3. **Attention pooling (instead of mean pooling)**
Learned attention weights over response tokens.

Result: Slightly worse performance (F1 ≈ 0.76) and higher risk of overfitting.

Reason: With only 689 training samples, the attention mechanism did not generalise well.
4. **LightGBM / Logistic Regression**
Replaced the MLP with classical classifiers.

Result: F1 ≤ 0.70.

Reason: Linear models cannot capture the non‑linear relationships between hidden states and hallucination patterns.

5. **No early stopping**

Training for fixed 300 epochs.

Result: Validation loss increased after ~100 epochs; test accuracy dropped by 2%.

Reason: Overfitting to training set noise.
# Final validation metric
Validation F1: 0.7758

Validation accuracy: 0.78

Best threshold: 0.30
