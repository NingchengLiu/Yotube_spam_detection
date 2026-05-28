# Hybrid LLM-ML Architecture for YouTube Spam Detection

This repository recreates the RSM8421 final project workflow for YouTube comment spam detection. The project combines lexical TF-IDF features with frozen RoBERTa sentence embeddings, then trains lightweight classical machine learning classifiers for efficient binary spam classification.

The target task is:

- `CLASS = 1`: spam comment
- `CLASS = 0`: non-spam comment

## Project Summary

YouTube comment spam often uses repeated promotional phrases, URLs, altered spelling, and self-promotion patterns such as "check my channel" or "subscribe". A pure rule-based detector is fast but brittle, while full LLM inference or transformer fine-tuning can be expensive for a small labeled dataset.

This project uses RoBERTa as a frozen feature extractor instead of a fine-tuned model. The semantic embeddings are concatenated with word-level and character-level TF-IDF features, then passed into classical classifiers.

## Reported Results

The class presentation reported the following final hybrid ensemble performance on an 80/20 stratified split of 1,956 YouTube comments:

| Metric | Score |
| --- | ---: |
| Accuracy | 97.2% |
| Precision | 98.0% |
| Recall | 96.5% |
| F1 score | 97.2% |
| ROC-AUC | 99.5% |

Feature ablation from the presentation:

| Feature set | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| TF-IDF only | 95.47% | 95.53% | 98.66% |
| RoBERTa only | 94.83% | 94.85% | 98.64% |
| Hybrid | 96.88% | 96.91% | 99.31% |

## Architecture

```text
comment text
  -> preprocessing
  -> word TF-IDF n-grams, 1-4, 5,000 features
  -> char TF-IDF n-grams, 2-6, 5,000 features
  -> frozen roberta-base mean-pooled embeddings, 768 features
  -> concatenate into 10,768-dimensional hybrid feature vector
  -> Logistic Regression / Random Forest / XGBoost / LightGBM
  -> optional soft-voting ensemble
```

The implementation keeps RoBERTa optional so the TF-IDF baseline can run on machines without PyTorch or HuggingFace Transformers.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- requirements-optional.txt
|-- scripts/
|   `-- download_dataset.py
|-- src/
|   `-- youtube_spam_detection/
|       |-- __init__.py
|       |-- features.py
|       |-- train.py
|       `-- utils.py
`-- tests/
    `-- test_utils.py
```

## Data

The project uses the KaggleHub dataset:

- Extended public dataset: https://www.kaggle.com/datasets/ahsenwaheed/youtube-comments-spam-dataset

Expected columns:

| Column | Description |
| --- | --- |
| `COMMENT_ID` | unique comment id |
| `AUTHOR` | commenter username |
| `DATE` | timestamp |
| `CONTENT` | comment text |
| `CLASS` | binary spam label |

The training CLI only requires `CONTENT` and `CLASS`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

For the TF-IDF baseline only, install:

```bash
python -m pip install pandas numpy scipy scikit-learn joblib
```

For hybrid RoBERTa features, also install PyTorch and Transformers:

```bash
python -m pip install -r requirements-optional.txt
```

## Download Data

```bash
python scripts/download_dataset.py
```

This uses:

```python
import kagglehub

path = kagglehub.dataset_download("ahsenwaheed/youtube-comments-spam-dataset")
```

The script combines all matching CSV files into:

```text
data/youtube_comments_spam.csv
```

## Usage

Run the TF-IDF baseline:

```bash
python -m youtube_spam_detection.train --data data/youtube_comments_spam.csv --feature-set tfidf
```

Run RoBERTa-only features:

```bash
python -m youtube_spam_detection.train --data data/youtube_comments_spam.csv --feature-set roberta
```

Run the hybrid feature model:

```bash
python -m youtube_spam_detection.train --data data/youtube_comments_spam.csv --feature-set hybrid
```

Save a fitted pipeline:

```bash
python -m youtube_spam_detection.train --data data/youtube_comments_spam.csv --feature-set tfidf --model-out artifacts/tfidf_model.joblib
```

## Model Choices

- TF-IDF captures exact spam vocabulary, URLs, n-grams, and misspellings.
- Character n-grams help detect spelling evasion such as `ch3ck`, `subscr1be`, or obfuscated links.
- RoBERTa embeddings capture semantic similarity and paraphrases that lexical features may miss.
- Classical classifiers keep inference efficient and interpretable compared with full transformer inference.

## Limitations

- The reported results are based on a relatively small YouTube spam collection centered on music video comments.
- Frozen RoBERTa embeddings are not adapted to the spam domain.
- Text-only classification ignores user history, posting frequency, video context, and network behavior.
- Spam tactics evolve, so production use would require monitoring and periodic retraining.

## Future Work

- Add user and channel behavior features.
- Fine-tune RoBERTa on a larger labeled spam corpus.
- Calibrate thresholds for moderation workflows with different false-positive costs.
- Add drift monitoring for new spam campaigns.
