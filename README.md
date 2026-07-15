# Financial Tweets Sentiment Analysis

This text mining project classifies financial tweets and short financial news snippets into **bearish**, **bullish**, or **neutral** sentiment. The final system used a finance-domain transformer encoder with an SVM classifier and achieved **0.98 macro F1** on the course test set, the best result in the course.

## Key Outcomes

- Built a full text classification workflow covering exploratory analysis, preprocessing, feature extraction, model comparison, and final test-set inference.
- Compared traditional sparse features, static word embeddings, transformer embeddings, fine-tuned transformer encoders, classifier heads, and zero-shot LLM classification.
- Selected **RashidNLP/Finance-Sentiment-Classification**, a finance-domain DeBERTa-based encoder, followed by an SVM classifier.
- Reached **97.20% average validation macro F1** for the best experiment configuration and **0.98 macro F1** on the test set.

## Methods and Tools

| Area | Used in this project |
| --- | --- |
| Text classification setup | Three-class financial sentiment classification for short tweets and news snippets |
| Text preprocessing | Ticker, URL, punctuation, casing, stop-word, stemming, and lemmatization experiments |
| Representation learning | TF-IDF, GloVe Twitter embeddings, finance-domain transformer embeddings |
| Models explored | Logistic Regression, KNN, Random Forest, SVM, fine-tuned transformer encoders, zero-shot LLM classification |
| Evaluation | Stratified validation, macro F1, per-class metrics |
| Python tools | pandas, NumPy, scikit-learn, NLTK, Hugging Face Transformers/Datasets, PyTorch, TensorFlow |

## Data and Target

The training set contains labeled financial text examples. The test set contains unlabeled examples used for final prediction.

| Split | Rows | Columns |
| --- | ---: | --- |
| Train | 9,543 | `text`, `label` |
| Test | 2,388 | `id`, `text` |

| Label | Sentiment | Train examples |
| ---: | --- | ---: |
| 0 | Bearish | 1,442 |
| 1 | Bullish | 1,923 |
| 2 | Neutral | 6,178 |

The main imbalance is the large Neutral class, which is roughly three times larger than the Bearish or Bullish classes.

## Main Challenges

| Challenge | Design choice |
| --- | --- |
| Short and noisy financial text | Inspected tickers, URLs, hashtags, mentions, punctuation patterns, numeric movements, and finance-specific symbols before choosing preprocessing strategies. |
| Imbalanced sentiment classes | Used macro F1 and stratified validation so Bearish and Bullish performance mattered alongside the majority Neutral class. |
| Representation choice | Compared TF-IDF, GloVe, multiple transformer encoders, fine-tuned encoders, and zero-shot Llama classification before selecting a finance-domain transformer embedding. |

## Workflow

<p align="center">
  <img width="800" alt="Experiments workflow" src="https://github.com/user-attachments/assets/82f28c37-1cbf-4c4c-9fb8-05774878c9c5" />
</p>

The workflow was organized into three notebooks:

1. Exploratory analysis and preprocessing
2. Preprocessing, embedding, and classifier experiments
3. Final model inference on the test set

## Experiment Results

The first experiment group compared preprocessing choices and feature extractors with logistic regression:

| Preprocessing | Feature Extractor | Classifier | F1-macro avg (%) |
| --- | --- | --- | ---: |
| Remove stop words, stemming, regex, lowercase, remove punctuation | TF-IDF <br><sub>`max_df=0.8`, `min_df=6`, `ngram_range=(1, 2)`</sub> | Logistic Regression | 71.75 |
| Remove stop words, lemmatization, regex, lowercase, remove punctuation | GloVe Twitter 200 | Logistic Regression | 62.68 |
| None | Transformer Encoder <br><sub>`zcharaf`</sub> | Logistic Regression | 92.60 |
| None | Transformer Encoder <br><sub>`HugMaik`</sub> | Logistic Regression | 87.62 |
| None | Transformer Encoder <br><sub>`RashidNLP`</sub> | Logistic Regression | **95.58** |

The top-performing embedder was then tested with different classifiers:

| Preprocessing | Embedder | Classifier | F1-macro avg (%) |
| --- | --- | --- | ---: |
| None | RashidNLP | Logistic Regression <br><sub>`penalty=l1`, `C=0.1`, `solver=liblinear`</sub> | 97.18 |
| None | RashidNLP | KNN <br><sub>`neighbors=15`, `weights=distance`</sub> | 96.78 |
| None | RashidNLP | Random Forest <br><sub>`n_estimators=100`, `max_depth=10`, `min_samples_leaf=2`</sub> | 96.91 |
| None | RashidNLP | SVM <br><sub>`C=10`, `kernel=rbf`, `gamma=scale`, `probability=True`</sub> | **97.20** |
| None | RashidNLP | Encoder head classifier | 86.59 |
| None | None | Llama-3.1-SuperNova-Lite <br><sub>zero-shot classification</sub> | 71.87 |

## Final Model

| Component | Selected configuration |
| --- | --- |
| Encoder | `RashidNLP/Finance-Sentiment-Classification` |
| Embedding strategy | CLS token embedding from the transformer encoder |
| Classifier | SVM |
| Best validation macro F1 | **97.20%** |
| Final test macro F1 | **0.98** |

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 00_eda_preprocessing.ipynb
│   ├── 01_modeling_experiments.ipynb
│   └── 02_final_submission.ipynb
└── src/
    ├── __init__.py
    └── utils.py
```

## Reproducing the Workflow

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebooks in numerical order. The final notebook writes predictions to:

```text
outputs/pred_21.csv
```

## Project Members

- Gaspar Pereira
- Iris Moreira
- Rafael Borges
- Rita Wang
