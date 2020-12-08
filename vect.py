from collections import Counter
from collections import defaultdict
import math
import copy


class Vectorizer:
    def __init__(self, preprocess, tokenize, stopwords=None, min_df=None, max_df=None,
                 max_features=None, vocabulary=None, idf=False):
        self._preprocess = preprocess
        self._tokenize = tokenize
        self._stopwords = stopwords
        self._min_df = min_df
        self._max_df = max_df
        self._max_features = max_features
        self._vocabulary = vocabulary
        self._vocabulary_ = defaultdict(tuple)
        self._idf = idf

    def fit(self, texts):
        docs = 0
        vocab = defaultdict(int)
        for text in texts:
            docs = docs + 1
            tokens = set(self._tokenize(self._preprocess(text)))
            for token in tokens:
                if self._vocabulary is not None:
                    if token in self._vocabulary:
                        vocab[token] += 1
                else:
                    vocab[token] += 1

        if self._max_features is not None:
            vocab = dict(Counter(vocab).most_common(self._max_features))

        copied_vocab = copy.deepcopy(vocab)
        for word, count in copied_vocab.items():
            if self._min_df is not None and count < self._min_df:
                del vocab[word]
            if self._max_df is not None and count > docs * self._max_df:
                del vocab[word]
        del copied_vocab

        for i, (word, count) in enumerate(vocab.items()):
            self._vocabulary_[word] = (i, math.log(float(docs + 1)/(count + 1))+1) if self._idf else (i, count)
        del vocab

    @property
    def vocabulary(self):
        return self._vocabulary_
