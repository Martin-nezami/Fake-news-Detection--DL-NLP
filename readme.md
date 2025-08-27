# Fake News Detection with NLP (Classical ML + Transformers)

## Overview
This project tackles **binary news classification**—distinguishing between *real* and *fake* news articles—using both **classical machine-learning and deep learning** methods and **modern transformer-based NLP**. The goal is to provide a clear, reproducible baseline and a stronger neural benchmark, so you can compare speed, interpretability, and accuracy across approaches in one place.

### What the notebook does

1. **Data Ingestion & Cleaning**
   Loads two labeled CSVs (commonly `True.csv` and `Fake.csv`), unifies their schema, and performs basic text cleanup (lowercasing, punctuation handling, and optional lemmatization/stopword removal with NLTK).

2. **Exploratory Data Analysis (EDA)**

   * Class distribution and text length summaries
   * Word frequency snapshots and optional **word clouds** for quick intuition

3. **Feature Engineering**

   * Classical track: **TF-IDF** vectorization (configurable n-gram range and vocabulary size)
   * Neural track: subword tokenization for transformer models (via `transformers`)

4. **Models Implemented**

   * **Classical ML (scikit-learn):**

     * Multinomial Naive Bayes
     * Logistic Regression
   * **Neural Models:**

     * **BERT** (via `transformers` + PyTorch) fine-tuned with a simple classification head
     * (Optional) **LSTM** in Keras/TensorFlow for a lightweight neural baseline

5. **Training & Evaluation Protocol**

   * Train/validation/test split (stratified)
   * Metrics: **Accuracy, Precision, Recall, F1, ROC-AUC**
   * Diagnostics: **Confusion matrix** and **ROC curve**

6. **Interpretability (Optional)**
   Uses **SHAP** to probe which terms or tokens influence predictions for classical models (and selected neural runs where feasible).

### Why this setup?

* **Baselines first:** Fast, interpretable classical models establish a strong reference.
* **Transformers second:** BERT typically improves performance on nuanced language, at higher compute cost.
* **Single notebook:** Everything—EDA, modeling, evaluation—is runnable end-to-end for easier replication and extension.

### How to use it

* Start with the **classical TF-IDF + Naive Bayes/LogReg** cells to get a quick, explainable baseline.
* Move to **BERT fine-tuning** if you need higher accuracy and can afford the compute.
* Swap in your dataset by updating the file paths and (if needed) the column names.
* Record your results (metrics and plots) and add them to the **Results** section of the README.


---
## Key Features

* **Data prep & EDA:** Class balance checks, text length distributions, and simple visualizations.
* **Text preprocessing:** Tokenization, stopword handling, optional lemmatization, TF-IDF vectorization.
* **Baselines:** Multinomial Naive Bayes, Logistic Regression, Linear SVM using `scikit-learn`.
* **Deep model:** BERT fine-tuning via `transformers` (with optional GPU acceleration).
* **Evaluation suite:** Accuracy, precision, recall, F1, ROC-AUC, confusion matrix; learning curves for deep runs.
* **Reproducibility:** Requirements file, environment setup guide, and random seed suggestions.


---
## System Requirements


**Python Version**
* Python 3.x (recommended **Python 3.7 or later**).

**Software Requirements**
**Python Packages**
Install with:
```bash
pip install numpy tensorflow matplotlib pillow scikit-learn
```
* `numpy`
* `tensorflow` (version 2.x)
* `matplotlib`
* `pillow` (PIL)
* `scikit-learn`

> For the full experiment set (including BERT and some visualizations), install from `requirements.txt` instead (see below).


---
## Project Structure

```
.
├── new project.ipynb       # Main notebook: EDA, preprocessing, models, evaluation
├── requirements.txt        # Python dependencies (full set, incl. transformers/torch)
└── README.md               # You are here
```


---

## Dataset
The notebook expects a labeled news dataset for binary classification (commonly the **True/Fake CSV** pair).
Update the file paths in the data-loading cells if your dataset lives elsewhere.



**Typical schema**
* `text` (news content/body)
* `label` (e.g., “FAKE” / “TRUE” or 0/1)

> You can substitute any similar dataset; just adapt the column names and preprocessing cells.

---

## Setup & Installation

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

**Option A — minimal (classical models only):**

```bash
pip install numpy tensorflow matplotlib pillow scikit-learn
```

**Option B — full stack (recommended):**

```bash
pip install -r requirements.txt
```

**(Optional) NLTK resources**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

> **GPU note:** If you have CUDA, install a GPU-enabled build of PyTorch following the selector on pytorch.org before running BERT experiments.

---

## How to Run

1. Launch Jupyter:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
2. Open `new project.ipynb`.
3. Run cells top-to-bottom. Adjust dataset paths if necessary.
4. (Optional) Toggle flags in the notebook to switch between classical models and BERT.

---

## Implementation Details

### Preprocessing

* Lowercasing, punctuation cleanup, optional lemmatization.
* Tokenization and stopword removal.
* TF-IDF features for classical models (configurable n-grams and max features).

### Classical Models (`scikit-learn`)

* **Multinomial Naive Bayes**
* **Logistic Regression**
* **Linear SVM**
* Hyperparameters set via grid search or simple defaults; see the notebook for details.

### Transformer Model (BERT)

* `transformers` + (PyTorch) fine-tuning with a simple classification head.
* Batch sizing & max sequence length configurable for memory constraints.
* Metrics tracked per epoch; early stopping is recommended on validation loss/F1.

---

## Evaluation & Reporting

* **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC.
* **Diagnostics:** Confusion matrix, ROC curve, learning curves (where applicable).
* **Recommendations:** Compare classical baselines vs BERT to weigh inference speed and accuracy. Consider calibration if probability estimates matter.

---

## Results & Visualizations

After running the notebook with your dataset, add:

* Final metric table (dev/test).
* Confusion matrix image.
* ROC curve plot.
* Any dataset-specific observations (e.g., class imbalance handling).

---

## Reproducibility Tips

* Fix seeds for `numpy`, `random`, and (if used) PyTorch.
* Pin versions in `requirements.txt` for long-term reproducibility.
* Log key configs (vectorizer settings, model hyperparameters, epochs).

---

## Contributing

Contributions are welcome!
Open an issue for bugs/ideas or submit a PR with improvements (new models, better preprocessing, more robust evaluation).

---

## License

Add your preferred license (e.g., **MIT**) here.

---

<sub>Structure and sectioning informed by the user’s reference document “Face Recognition Using Meta-Learning.”</sub>&#x20;
