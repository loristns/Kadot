from .tokenizers import RegexTokenizer
from .fuzz import most_similar


class BaseCorrector(object):

    def __init__(self, tokenizer=RegexTokenizer()):
        self.tokenizer = tokenizer

    def fit(self, documents):
        pass

    def fit_from_file(self, filename):
        with open(filename) as document_file:
            self.fit([document_file.read()])

    def predict(self, documents):
        pass


class NaiveCorrector(BaseCorrector):
    """A simple naive spell checker"""

    def __init__(self, tokenizer=RegexTokenizer()):
        BaseCorrector.__init__(self, tokenizer=tokenizer)
        self.words = set()

    def fit(self, documents):
        for document in documents:
            self.words.update(self.tokenizer.tokenize(document))
            self.words.update(self.tokenizer.tokenize(document.lower()))  # Add a lowercase version of the vocabulary

    def predict(self, documents):
        new_documents = []

        for document in documents:
            document = self.tokenizer.tokenize(document)
            corrected_document = document.copy()

            for word_position, word in enumerate(document):
                if word not in self.words:
                    corrected_document[word_position] = most_similar(word, self.words)[0][0]

            new_documents.append(self.tokenizer.rebuild_last(corrected_document))

        return new_documents
