from .tokenizers import RegexTokenizer
from .vectorizers import DocVectorizer
from collections import OrderedDict
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class BaseClassifier(object):

    def __init__(self, tokenizer=RegexTokenizer(), vectorizer=DocVectorizer()):
        self.vectorizer = vectorizer
        self.vectorizer.tokenizer = tokenizer

    def fit(self, documents):
        """
        :param documents: a dict containing text as keys and label as values
        """

        documents = OrderedDict(documents)

        self.labels = list(documents.values())
        self.text_vectors = self.vectorizer.fit_transform(list(documents.keys()))

    def predict(self, documents):
        """
        :param documents: a list of document to classify.
        """
        pass


class SVMClassifier(BaseClassifier):

    def fit(self, documents):
        BaseClassifier.fit(self, documents)

        self.sk_model = SVC().fit(self.text_vectors.values(), self.labels)

    def predict(self, documents):
        if not isinstance(documents, list):
            documents = [str(documents)]

        unique_word_save = self.vectorizer.unique_words
        self.vectorizer.fit(documents)
        self.vectorizer.unique_words = unique_word_save

        predict_vectors = self.vectorizer.transform()

        return dict(zip(documents, self.sk_model.predict(predict_vectors.values())))


class BayesClassifier(SVMClassifier):

    def fit(self, documents):
        BaseClassifier.fit(self, documents)

        self.sk_model = GaussianNB().fit(self.text_vectors.values(), self.labels)
