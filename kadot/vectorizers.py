from .tokenizers import CharTokenizer
from .core import VectorDictionary, VectorCoordinate
from statistics import mean


class BaseVectorizer(object):
    """
    A base class for building vectorizers.
    """

    def __init__(self, tokenizer=CharTokenizer()):
        """
        :param tokenizer: A BaseTokenizer subclass object to tokenize the text
        """

        self.tokenizer = tokenizer

        self.documents = []  # List of raw documents
        self.documents_corpus = []  # Tokenized lowercased concatenation of all documents
        self.unique_words = []  # List of uniques words in `self.document_corpus`

    def fit(self, documents):
        if isinstance(documents, list):
            self.documents += documents
        else:
            self.documents.append(documents)

        self.documents_corpus = self.tokenizer.tokenize(" ".join(self.documents).lower())
        self.unique_words = list(set(self.documents_corpus))

    def transform(self, window):
        pass

    def fit_transform(self, documents, window):
        self.fit(documents)
        return self.transform(window)


class WordVectorizer(BaseVectorizer):
    """
    A simple distributional vectorizer algorithm.
    """

    def transform(self, window):
        vector_dict = VectorDictionary(dimension=len(self.unique_words))

        for n_word in self.unique_words:
            n_word_indexes = [i for i, x in enumerate(self.documents_corpus) if x == n_word]  # list of index of `n_word`
            n_word_vectors = []

            if n_word_indexes:
                # Build a vector for each index...
                for index in n_word_indexes:
                    text_selection = self.documents_corpus[index - window:index] +\
                                     self.documents_corpus[index + 1:index + window + 1]
                    n_word_vectors.append([text_selection.count(word) for word in self.unique_words])

                # ...And mean them to build the final vector :
                vector_dict[n_word] = VectorCoordinate(map(mean, zip(*n_word_vectors)))

            else:
                vector_dict[n_word] = [0 for i in range(len(self.unique_words))]

        return vector_dict


class DocVectorizer(BaseVectorizer):
    """
    Use WordVectorizer to vectorize a whole text.
    """

    def transform(self, window):
        vector_dict = VectorDictionary(dimension=len(self.unique_words))

        for document in self.documents:
            vectorizer = WordVectorizer(tokenizer=self.tokenizer)
            vectorizer.fit(document)
            vectorizer.unique_words = self.unique_words

            document_vocabulary_vectors = vectorizer.transform(window)
            vector_dict[document] = VectorCoordinate(map(mean, zip(*document_vocabulary_vectors.values())))

        return vector_dict
