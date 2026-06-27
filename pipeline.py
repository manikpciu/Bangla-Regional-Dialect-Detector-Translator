import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class RegionalLexiconFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, regional_lexicon, text_col='text_clean'):
        self.regional_lexicon = regional_lexicon
        self.text_col = text_col
        self.regions = sorted(regional_lexicon.keys())

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        features = np.zeros((len(X), len(self.regions) * 2))

        for i, text in enumerate(X):

            text_words = set(str(text).split())

            for j, region in enumerate(self.regions):

                region_words = set(
                    self.regional_lexicon.get(region, [])
                )

                count = len(
                    text_words.intersection(region_words)
                )

                features[i, j * 2] = count

                features[i, j * 2 + 1] = 1 if count > 0 else 0

        return csr_matrix(features)