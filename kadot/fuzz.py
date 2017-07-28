"""
Freely inspired by FuzzyWuzzy: https://github.com/seatgeek/fuzzywuzzy
"""

from .tokenizers import RegexTokenizer
from collections import Counter
from difflib import SequenceMatcher


def ratio(string_a, string_b, ignore_case=False):
    """
    Compute the similarity ratio between two string.

    Examples
    --------
    >>> ratio('Hola', 'Hello')
    0.444
    """

    if ignore_case:
        string_a = string_a.lower()
        string_b = string_b.lower()

    return round(SequenceMatcher(a=string_a, b=string_b).ratio(), 3)


def vocabulary_ratio(string_a, string_b, ignore_case=False, tokenizer=RegexTokenizer()):
    """
    Get the vocabulary of two string and compute the ratio between them.

    Examples
    --------
    >>> vocabulary_ratio('This is a text !', 'This is a-text !')
    1.0
    """

    vocabulary_a = " ".join(sorted(set(tokenizer.tokenize(str(string_a)))))
    vocabulary_b = " ".join(sorted(set(tokenizer.tokenize(str(string_b)))))

    return ratio(vocabulary_a, vocabulary_b, ignore_case)


def most_similar(query, choices, best=None, ignore_case=False, ratio_function=ratio):
    """
    Return the `best` most similar `choices` with `query`. Useful for searching.

    Examples
    --------
    >>> most_similar("San Fr", ['New-York City', 'San Francisco', 'Los Angeles'], best=1)
    [('San Francisco', 0.632)]
    """

    similarity_dict = dict()

    for choice in choices:
        similarity_dict[choice] = ratio_function(query, choice, ignore_case)

    return Counter(similarity_dict).most_common(best)


class BaseCorrector(object):

    def __init__(self, tokenizer=RegexTokenizer()):
        self.tokenizer = tokenizer

    def fit(self, documents):
        pass

    def predict(self, documents):
        pass


class NaiveCorrector(BaseCorrector):
    """A simple naive spell checker"""

    def __init__(self, tokenizer=RegexTokenizer()):
        BaseCorrector.__init__(tokenizer)
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
