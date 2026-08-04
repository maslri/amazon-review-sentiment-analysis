# Dataset

The dataset used in this project is not included in this repository because of its size.

## Dataset Source

The project uses the Amazon Product Reviews dataset.

## Folder Structure

Place the dataset files inside:

```
data/raw/
```

Expected files:

```
test_data.csv
title_brand.csv
train_data.csv
```

## Processed datasets

The complete preprocessing workflow is in:

```text
notebooks/02_preprocessing.ipynb
```

The notebook removes exact duplicates, creates a balanced sample, applies
light preprocessing suitable for BERT-family models, and creates a stratified
train-validation split.

The processed files include a selected `model_input` column with this format:

```text
Summary: ... | Verified: yes/no | Helpful votes: bucket | Review: ...
```

Product title, brand, style, dates, and identifier fields were considered and
audited. Title and brand remain available as separate columns, but they are not
injected into `model_input` because validation ablation reduced micro F1 and
their added tokens reduce the review context available to BERT-family models.

The experiment size is controlled by one value:

```python
SAMPLES_PER_CLASS = 10_000
```

Changing it to `50_000` and rerunning the notebook creates a new isolated
output directory without overwriting the 10,000-per-class experiment.

Current outputs are stored under:

```text
data/processed/bert_balanced_10000_per_class/
├── balanced_reviews.csv
├── train.csv
├── validation.csv
└── metadata.json
```

The official raw test dataset is never included in preprocessing, splitting,
or model selection.

For Kaggle notebooks, the dataset will be loaded directly from the Kaggle input directory.
