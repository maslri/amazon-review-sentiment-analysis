from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.config import IS_KAGGLE
from src.config import KAGGLE_INPUT


def load_reviews():

    if IS_KAGGLE:
        path = KAGGLE_INPUT / "YOUR_DATASET_NAME" / "reviews_train.csv"
    else:
        path = RAW_DATA_DIR / "reviews_train.csv"

    return pd.read_csv(path)


def load_products():

    if IS_KAGGLE:
        path = KAGGLE_INPUT / "YOUR_DATASET_NAME" / "products.csv"
    else:
        path = RAW_DATA_DIR / "products.csv"

    return pd.read_csv(path)
