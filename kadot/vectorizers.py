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

        self.raw_documents = []  # List of raw documents
        self.processed_documents = []  # List of tokenized, lowercased documents
        self.unique_words = []  # List of uniques words in `self.document_corpus`

    def fit(self, documents):
        if isinstance(documents, list):
            self.raw_documents += documents
            for doc in documents:
                self.processed_documents.append(self.tokenizer.tokenize(doc.lower()))

        else:
            self.raw_documents.append(documents)
            self.processed_documents.append(self.tokenizer.tokenize(documents.lower()))

        documents_corpus = self.tokenizer.tokenize(" ".join(self.raw_documents).lower())
        self.unique_words = list(set(documents_corpus))

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
            n_word_vectors = []

            for document in self.processed_documents:
                doc_n_word_indexes = [i for i, x in enumerate(document) if x == n_word]  # list of index of `n_word` in document

                if doc_n_word_indexes:
                    # Build a vector for each index...
                    for index in doc_n_word_indexes:
                        text_selection = document[index - window:index] +\
                                         document[index + 1:index + window + 1]
                        n_word_vectors.append([text_selection.count(word) for word in self.unique_words])

                else:
                    n_word_vectors.append([0 for i in self.unique_words])

            # ...And mean them to build the final vector :
            vector_dict[n_word] = VectorCoordinate(map(mean, zip(*n_word_vectors)))

        return vector_dict


class DocVectorizer(BaseVectorizer):
    """
    Use WordVectorizer to vectorize a whole text.
    """

    def transform(self, window):
        vector_dict = VectorDictionary(dimension=len(self.unique_words))

        for document in self.raw_documents:
            vectorizer = WordVectorizer(tokenizer=self.tokenizer)
            vectorizer.fit(document)
            vectorizer.unique_words = self.unique_words

            document_vocabulary_vectors = vectorizer.transform(window)
            vector_dict[document] = VectorCoordinate(map(mean, zip(*document_vocabulary_vectors.values())))

        return vector_dict
