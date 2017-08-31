from .tokenizers import RegexTokenizer
from .vectorizers import DocVectorizer
from collections import OrderedDict


class BaseClassifier(object):

    def __init__(self, tokenizer=RegexTokenizer(), vectorizer=DocVectorizer()):
        self.vectorizer = vectorizer
        self.vectorizer.tokenizer = tokenizer

        self.labels = []
        self.text_vectors = {}

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


class ScikitClassifier(BaseClassifier):

    def __init__(self, scikit_classifier, tokenizer=RegexTokenizer(), vectorizer=DocVectorizer()):
        BaseClassifier.__init__(self, tokenizer, vectorizer)

        self.scikit_model = scikit_classifier

    def fit(self, documents):
        BaseClassifier.fit(self, documents)

        self.scikit_model.fit(self.text_vectors.values(), self.labels)

    def predict(self, documents):
        if not isinstance(documents, list):
            documents = [str(documents)]

        unique_word_save = self.vectorizer.unique_words
        self.vectorizer.fit(documents)
        self.vectorizer.unique_words = unique_word_save

        predict_vectors = self.vectorizer.transform()

        return dict(zip(documents, self.scikit_model.predict(predict_vectors.values())))

