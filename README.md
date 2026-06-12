# Amazon Sentiment Analysis

Sentiment analysis on the Amazon Reviews dataset (**Cell Phones & Accessories** category). This project compares a **classical machine learning baseline**, two **deep learning (LSTM)** approaches, and a fine-tuned **Transformer (DistilBERT)** model, and ships an interactive **Streamlit** app for live inference.

---

## Overview

The pipeline takes raw Amazon reviews, converts star ratings into sentiment labels, performs EDA, generates model-specific preprocessed text, and trains/evaluates four families of models:

| Model | Type | Task | Key Features |
|-------|------|------|--------------|
| TF-IDF + Logistic Regression | Classical ML | 5-class | `RandomizedSearchCV` hyperparameter tuning, `f1_macro` scoring |
| BiLSTM | Deep Learning | 5-class | Learned embeddings, class-weighted loss, early stopping |
| BiLSTM + Attention | Deep Learning | Binary | Pretrained GloVe embeddings, attention mechanism, label smoothing |
| DistilBERT | Transformer | Binary | Fine-tuned, weighted loss for imbalance, mixed-precision (fp16) |

---

## Dataset & Labeling

Reviews are loaded from a JSONL file, extracting `reviewText` and the numeric `overall` rating (1–5).

**5-class mapping** (used in EDA, classical baseline, and the multiclass LSTM):

| Rating | Sentiment |
|--------|-----------|
| 1 | very_negative |
| 2 | negative |
| 3 | neutral |
| 4 | positive |
| 5 | very_positive |

**Binary mapping** (used in the attention LSTM and DistilBERT):

| Sentiment | Label |
|-----------|-------|
| very_negative, negative, neutral | 0 (Negative) |
| positive, very_positive | 1 (Positive) |

Data is split into **train / validation / test (80 / 10 / 10)** using **stratified** sampling to preserve class balance.

---

## Exploratory Data Analysis (`data_exploration.ipynb`)

- Class distribution (counts and percentages) across the five sentiment categories
- Review length analysis by word count, including a log-transform to address right skew
- Outlier detection via the IQR method on review length — flagged outliers were **retained**, as they represent genuine long-form reviews rather than corrupt data
- Stratified train/val/test split, saved to `data/processed/`

---

## Preprocessing (`prepocessing.ipynb`)

Three preprocessed text columns are generated, one tuned for each model family:

- **`text_ml`** — full cleaning for classical ML: lowercasing, URL/HTML removal, tokenization, non-alphabetic removal, **negation handling** (`not good` → `not_good`), stopword removal, and lemmatization
- **`text_dl`** — light cleaning for deep learning: lowercasing, URL/HTML/whitespace normalization only (preserves word order and structure for the LSTMs)
- **`text_transformer`** — light cleaning for DistilBERT (the model's own tokenizer handles the rest)

Output is saved as `*_all_versions.csv` for each split.

---

## Models

### 1. Classical Baseline (`classical_baseline.ipynb`)
- **TF-IDF + Logistic Regression** pipeline
- `RandomizedSearchCV` (5-fold CV) over n-gram range, vocabulary size, `min_df`/`max_df`, regularization strength `C`, class weighting, and solver
- Scored on `f1_macro` to handle class imbalance
- Evaluated with accuracy, macro F1/precision/recall, classification reports, and confusion matrices
- Best model saved to `models/tfidf_logreg_best_model.pkl`

### 2. BiLSTM — Multiclass (`lstm.ipynb`)
- Bidirectional LSTM (2 layers, 256 hidden units, 300-dim learned embeddings)
- Custom vocabulary (min frequency 3), sequences padded to length 200
- Class-weighted `CrossEntropyLoss`, gradient clipping, early stopping on validation loss
- Best checkpoint saved as `best_bilstm_model.pt`

### 3. BiLSTM + Attention — Binary (`lstm_2labels.ipynb`)
- Bidirectional LSTM with an **attention mechanism** over time steps
- **Pretrained GloVe** embeddings (`glove.6B.300d`), frozen initially then unfrozen for fine-tuning
- Class-weighted loss with label smoothing, `ReduceLROnPlateau` scheduler, early stopping
- Loss/accuracy curves and confusion matrix for evaluation
- Best checkpoint saved as `best_binary_attention.pt`

### 4. DistilBERT — Binary (`transformers.ipynb`)
- Fine-tuned `distilbert-base-uncased` via the Hugging Face `Trainer`
- Custom `WeightedTrainer` overrides the loss to apply class weights for imbalance
- Mixed-precision (fp16), gradient accumulation (effective batch size 32), warmup, weight decay, early stopping
- Best checkpoint saved under `results/checkpoint-*`

---

## Streamlit App (`app.py`)

An interactive demo that loads the fine-tuned DistilBERT checkpoint and predicts sentiment on user-entered review text, displaying the prediction and confidence score.

```bash
streamlit run app.py
```

> **Note:** Update `CHECKPOINT_PATH` in `app.py` to point to your local DistilBERT checkpoint directory.

---

## Project Structure

```
amazon-sentiment-analysis/
├── data_raw/
│   └── reviews.json                  # Raw Amazon reviews (JSONL)
├── data/
│   ├── processed/                    # Train/val/test splits + all-versions CSVs
│   └── glove/
│       └── glove.6B.300d.txt         # Pretrained GloVe embeddings
├── models/
│   └── tfidf_logreg_best_model.pkl
├── notebooks/
│   ├── data_exploration.ipynb        # EDA + train/val/test split
│   ├── prepocessing.ipynb            # Text cleaning (ML / DL / Transformer variants)
│   ├── classical_baseline.ipynb      # TF-IDF + Logistic Regression
│   ├── lstm.ipynb                    # BiLSTM (5-class)
│   ├── lstm_2labels.ipynb            # BiLSTM + Attention + GloVe (binary)
│   ├── transformers.ipynb            # DistilBERT fine-tuning (binary)
│   └── results/                      # DistilBERT checkpoints
└── app.py                            # Streamlit inference app
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key dependencies
`pandas` · `numpy` · `scikit-learn` · `nltk` · `torch` · `transformers` · `matplotlib` · `seaborn` · `joblib` · `streamlit` · `tqdm`

NLTK resources (downloaded inside the notebooks):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Usage

1. Place the raw dataset at `data_raw/reviews.json`.
2. Run `notebooks/data_exploration.ipynb` to perform EDA and create the splits.
3. Run `notebooks/prepocessing.ipynb` to generate the model-specific text columns.
4. Run any of the model notebooks (`classical_baseline`, `lstm`, `lstm_2labels`, `transformers`) to train and evaluate.
5. Launch the demo with `streamlit run app.py`.

> A CUDA-capable GPU is recommended for the LSTM and DistilBERT notebooks. They fall back to CPU automatically if no GPU is available.

---

## Notes on Class Imbalance

The dataset is skewed toward positive reviews. Every model accounts for this:
- **Classical / LSTM / DistilBERT:** class-weighted loss (or `class_weight="balanced"`)
- **Evaluation:** macro-averaged F1 to weight all classes equally rather than rewarding majority-class predictions