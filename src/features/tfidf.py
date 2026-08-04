from pathlib import Path

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFVectorizer:
    """
    Wrapper around sklearn's TfidfVectorizer.
    """

    def __init__(
        self,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    ):

        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            lowercase=lowercase,
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    @property
    def feature_names(self):
        return self.vectorizer.get_feature_names_out()

    @property
    def feature_count(self):
        return len(self.feature_names)

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.vectorizer,
            path,
        )

    @classmethod
    def load(cls, path):

        obj = cls()

        obj.vectorizer = joblib.load(path)

        return obj
