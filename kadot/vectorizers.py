from .tokenizers import CharTokenizer
from .core import VectorDictionary
import numpy as np


class BaseVectorizer(object):
    """
    A base class for building vectorizers with a scikit-learn like API.
    """

    def __init__(self, window=5, tokenizer=CharTokenizer()):
        """
        :param tokenizer: A BaseTokenizer subclass object to tokenize the text
        """

        self.tokenizer = tokenizer
        self.window = window

        self.unique_words = []  # List of uniques words in `documents_corpus`, see lower.

    def fit(self, documents):
        self.raw_documents = []  # List of raw documents
        self.processed_documents = []  # List of tokenized, lowercased documents

        if isinstance(documents, list):
            self.raw_documents += documents
            for doc in documents:
                self.processed_documents.append(self.tokenizer.tokenize(doc.lower()))

        else:
            self.raw_documents.append(documents)
            self.processed_documents.append(self.tokenizer.tokenize(documents.lower()))

        documents_corpus = self.tokenizer.tokenize(" ".join(self.raw_documents).lower())
        self.unique_words = list(set(self.unique_words) | set(documents_corpus))

    def transform(self):
        pass

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform()

    def predict(self, documents):
        pass


class WordVectorizer(BaseVectorizer):
    """
    A simple distributional vectorizer algorithm.
    """

    def transform(self):
        vector_dict = VectorDictionary(dimension=len(self.unique_words))

        for n_word in self.unique_words:
            n_word_vectors = np.empty([0, len(self.unique_words)], dtype=int)

            for document in self.processed_documents:
                doc_n_word_indexes = [i for i, x in enumerate(document) if x == n_word]  # list of index of `n_word` in document

                if doc_n_word_indexes:
                    # Build a vector for each index...
                    for index in doc_n_word_indexes:
                        text_selection = document[index - self.window:index] +\
                                         document[index + 1:index + self.window + 1]

                        n_word_vectors = np.append(n_word_vectors,
                                                   np.array([[text_selection.count(word) for word in self.unique_words]]),
                                                   axis=0)

                else:
                    np.append(n_word_vectors, np.zeros(len(self.unique_words)))

            # ...And sum them to build the final vector :
            vector_dict[n_word] = np.sum(n_word_vectors, axis=0)

        return vector_dict


class DocVectorizer(BaseVectorizer):
    """
    A simple count based/bag-of-word vectorizer, to vectorize a whole text.
    """

    def transform(self):
        vector_dict = VectorDictionary(dimension=len(self.unique_words))

        for raw_document, processed_document in zip(self.raw_documents, self.processed_documents):
            vector_dict[raw_document] = np.array([processed_document.count(word) for word in self.unique_words], dtype=int)

        return vector_dict


class SemanticDocVectorizer(BaseVectorizer):
    """
    Use the semantic WordVectorizer to vectorize a whole text.
    """

    def transform(self):
        vector_dict = VectorDictionary(dimension=len(self.unique_words))

        for document in self.raw_documents:
            vectorizer = WordVectorizer(window=self.window, tokenizer=self.tokenizer)
            vectorizer.fit(document)
            vectorizer.unique_words = self.unique_words

            document_vocabulary_vectors = vectorizer.transform()
            vector_dict[document] = np.mean(np.array(document_vocabulary_vectors.values()), axis=0)

        return vector_dict
