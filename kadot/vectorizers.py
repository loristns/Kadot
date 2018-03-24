from kadot.tokenizers import Tokens
from kadot.utils import SavedObject, unique_words
from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.sparse, scipy.spatial

CBOW_MODEL = {'sg': 0, 'min_count': 1}
SKIP_GRAM_MODEL = {'sg': 1, 'min_count': 1}


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculates the cosine similarity of two vectors.
    """

    # 1 - distance = similarity
    return 1 - scipy.spatial.distance.cosine(vector1, vector2)


def handle_corpus(corpus: Union[Tokens, Sequence[Tokens]]) -> Tuple[List[List[str]], List[str]]:
    """
    Utility function to manage the `corpus` argument of a vectorizer.

    :param corpus: a Tokens object or an iterable object containing
     Tokens objects.

    :return: a tuple containing the list of tokenized texts (in a list form)
     and the list of raw texts.
    """

    if isinstance(corpus, Tokens):
        corpus = [corpus]

    tokens = []
    raw_texts = []
    for text in corpus:
        tokens.append(text.tokens)
        raw_texts.append(text.raw)

    return tokens, raw_texts


class VectorDict(SavedObject):
    """
    An ordered dict to store vectors associated with their documents.
    """

    def __init__(self,
                 keys: list,
                 matrix: Union[np.ndarray, scipy.sparse.lil_matrix],
                 sparse=False):

        self.vocabulary = keys
        self.is_sparse = sparse

        if self.is_sparse:
            self.matrix = scipy.sparse.lil_matrix(matrix)
        else:
            self.matrix = matrix

    def __repr__(self):
        return "VectorDict({})".format(self.matrix.shape)

    def __getitem__(self, item):
        if isinstance(item, str):
            item_index = self.vocabulary.index(item)
        elif isinstance(item, int):
            item_index = item

        if self.is_sparse:
            return self.matrix[item_index].toarray()
        else:
            return self.matrix[item_index]

    def keys(self):
        return self.vocabulary

    def values(self):
        return self.matrix

    def items(self):
        return list(zip(self.keys(), self.values()))

    def g_values(self):
        """
        A generator behaving like the `value` method but converting
        sparse matrices into numpy array.
        """

        for value in self.matrix:
            if self.is_sparse:
                yield value.toarray()
            else:
                yield value

    def g_items(self):
        """
        A generator behaving like the `item` method but converting
        sparse matrices into numpy array.
        """

        for key, value in self.items():
            if self.is_sparse:
                yield key, value.toarray()
            else:
                yield key, value

    def most_similar(self,
                     coordinates: Union[str, np.ndarray],
                     best: Optional[int] = 10,
                     exclude: Optional[Sequence[str]] = None,
                     similarity: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity
                     ) -> List[Tuple[str, float]]:
        """
        Find the `best` most similar words to `coordinates'.

        :param coordinates: a vector (numpy array) or a key of a vector
         in the dictionary.

        :param exclude: a list of words to exclude from the result.

        :param similarity: a function calculating the similarity between two vectors.

        :return: a list of tuples containing the most similar words
         and their cosine similarities.
        """

        if not isinstance(coordinates, np.ndarray):
            coordinates = self[coordinates]

        if exclude is None:
            exclude = []

        similarity_dict = dict()

        for key, key_coordinates in self.g_items():
            if key not in exclude:
                similarity_dict[key] = similarity(key_coordinates, coordinates)

        return Counter(similarity_dict).most_common(best)

    def analogy(self,
                from1: Union[str, np.ndarray],
                to1: Union[str, np.ndarray],
                from2: Union[str, np.ndarray],
                best: int = 10,
                exclude: Optional[Sequence[str]] = None,
                similarity: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity
                ) -> List[Tuple[str, float]]:
        """
        Find the words that match the most closely to an analogy.
        For example: "The man is to the woman what the king is to... the queen."
        """

        if not isinstance(from1, np.ndarray): from1 = self[from1]
        if not isinstance(from2, np.ndarray): from2 = self[from2]
        if not isinstance(to1, np.ndarray): to1 = self[to1]

        return self.most_similar(to1 - from1 + from2, best, exclude, similarity)

    def doesnt_match(self, keys: list) -> str:
        """
        Search the intruder in a list of keys.
        """

        average_vector = np.mean([self[key] for key in keys], axis=0)

        max_distance = 0
        max_distance_key = None
        for key in keys:
            vector = self[key]
            vector_distance = scipy.spatial.distance.cosine(average_vector,
                                                            vector)
            if vector_distance > max_distance:
                max_distance = vector_distance
                max_distance_key = key

        return max_distance_key


def word_vectorizer(
        corpus: Union[Tokens, Sequence[Tokens]],
        window: int = 4,
        vocabulary: Optional[Sequence[str]] = None
        ) -> VectorDict:
    """
    A distributional word vectorizer constructing a co-occurrence matrix
    and applying a dimension reduction algorithm on it.

    :param corpus: a list of Tokens objects or a Tokens object.

    :param window: the size of a word context window side.

    :param vocabulary: the vocabulary used for vectorization in the form of a
     word list, or if None (default), the vocabulary would be extracted from the corpus.

    :return: a VectorDict object containing the word vectors.
    """

    corpus_tokens, corpus_texts = handle_corpus(corpus)

    if vocabulary is None:
        vocabulary = unique_words(sum(corpus_tokens, []))

    len_vocabulary = len(vocabulary)
    cooc_matrix = scipy.sparse.lil_matrix((len_vocabulary, len_vocabulary))

    for document_tokens in corpus_tokens:

        len_document = len(document_tokens)
        doc_voc_indices = [vocabulary.index(word) for word in document_tokens]

        for word_position, row_index in enumerate(doc_voc_indices):
            word_window = doc_voc_indices[max(0, word_position - window): word_position] +\
                          doc_voc_indices[word_position + 1: min(len_document + 1, word_position + 1 + window)]

            for col_index in word_window:
                    cooc_matrix[row_index, col_index] += 1

    return VectorDict(vocabulary, cooc_matrix, sparse=True)


def word2vec_vectorizer(
        corpus: Union[Tokens, Sequence[Tokens]],
        dimension: int,
        window: int = 4,
        iter: int = 1000,
        model=CBOW_MODEL
        ) -> VectorDict:
    """
    A word vectorizer using the word2vec algorithm (require Gensim).

    :param corpus: a list of Tokens objects or a Tokens object.

    :param dimension: the number of dimensions of word vectors.

    :param model: parameters to pass to the gensim vectorizer.
     https://radimrehurek.com/gensim/models/word2vec.html

    :return: a VectorDict object containing the word vectors.
    """

    from gensim.models import Word2Vec

    corpus_tokens, _ = handle_corpus(corpus)
    vocabulary = unique_words(sum(corpus_tokens, []))

    vector_matrix = np.zeros((len(vocabulary), dimension))
    word2vec = Word2Vec(corpus_tokens, size=dimension, window=window,
                        iter=iter, **model)

    for row_index, word in enumerate(vocabulary):
        vector_matrix[row_index] = word2vec.wv[word]

    return VectorDict(vocabulary, vector_matrix)


def fasttext_vectorizer(
        corpus: Union[Tokens, Sequence[Tokens]],
        dimension: int,
        window: int = 4,
        iter: int = 1000,
        model=CBOW_MODEL
        ) -> VectorDict:
    """
    A word vectorizer using the FastText algorithm (require Gensim).

    :param corpus: a list of Tokens objects or a Tokens object.

    :param dimension: the number of dimensions of word vectors.

    :param model: parameters to pass to the gensim vectorizer.
     https://radimrehurek.com/gensim/models/fasttext.html

    :return: a VectorDict object containing the word vectors.
    """

    from gensim.models.fasttext import FastText

    corpus_tokens, _ = handle_corpus(corpus)
    vocabulary = unique_words(sum(corpus_tokens, []))

    vector_matrix = np.zeros((len(vocabulary), dimension))
    word2vec = FastText(corpus_tokens, size=dimension, window=window,
                        iter=iter, **model)

    for row_index, word in enumerate(vocabulary):
        vector_matrix[row_index] = word2vec.wv[word]

    return VectorDict(vocabulary, vector_matrix)


def centroid_document_vectorizer(
        corpus: Union[Tokens, Sequence[Tokens]],
        word_vectors: VectorDict,
        sparse = False
        ) -> VectorDict:
    """
    A document vectorizer calculating the centroid
    of the vectors of the words it contains.

    :param corpus: a list of Tokens objects or a Tokens object.

    :param word_vectors: a VectorDict object containing the word vectors.

    :return: a VectorDict object containing the document vectors.
    """

    corpus_tokens, corpus_texts = handle_corpus(corpus)

    vector_matrix = np.zeros((len(corpus_texts), word_vectors.values().shape[1]))
    if sparse:
        vector_matrix = scipy.sparse.lil_matrix(vector_matrix)

    for row_index, document_tokens in enumerate(corpus_tokens):
        for token in document_tokens:
            if token in word_vectors.keys():
                vector_matrix[row_index] += word_vectors[token] / len(document_tokens)

    return VectorDict(corpus_texts, vector_matrix, sparse)


def count_document_vectorizer(
        corpus: Union[Tokens, Sequence[Tokens]],
        vocabulary: Optional[Sequence[str]] = None
        ) -> VectorDict:
    """
    A simple "bag-of-words" document vectorizer that simply counts the number
    of occurrences of each word in each document.

    :param corpus: a list of Tokens objects or a Tokens object.

    :param vocabulary: the vocabulary used for vectorization in the form of a
     word list, or if None (default), the vocabulary would be extracted from the corpus.

    :return: a VectorDict object containing the document vectors.
    """

    corpus_tokens, corpus_texts = handle_corpus(corpus)

    if vocabulary is None:
        vocabulary = unique_words(sum(corpus_tokens, []))

    count_matrix = scipy.sparse.lil_matrix((len(corpus_texts), len(vocabulary)))

    for row_index, document_tokens in enumerate(corpus_tokens):
        doc_voc_indices = [vocabulary.index(word) for word in document_tokens if word in vocabulary]

        for col_index in doc_voc_indices:
            count_matrix[row_index, col_index] += 1

    return VectorDict(corpus_texts, count_matrix, sparse=True)
