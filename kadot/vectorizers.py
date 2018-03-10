from kadot.tokenizers import Tokens
from kadot.utils import SavedObject, unique_words
from collections import Counter, OrderedDict, UserDict
from typing import Callable, List, Optional, Sequence, Tuple, Union
from gensim.models import Word2Vec  # TODO : Switch to pytorch
import numpy as np
import scipy.spatial.distance
from sklearn.decomposition import TruncatedSVD

CBOW_MODEL = {'sg': 0}
SKIP_GRAM_MODEL = {'sg': 1}


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


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculates the cosine similarity of two vectors.
    """

    # 1 - distance = similarity
    return 1 - scipy.spatial.distance.cosine(vector1, vector2)


class VectorDict(SavedObject, UserDict):
    """
    An ordered dict to store vectors associated with their documents.
    """

    def __init__(self, *args, **kwargs):
        UserDict.__init__(self)

        self.data = OrderedDict(*args, **kwargs)
        self.unique_words = None

    def __repr__(self):
        return "VectorDict()"

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[self.keys()[item]]
        else:
            return self.data[item]

    def keys(self):
        return list(self.data.keys())

    def values(self):
        return list(self.data.values())

    def items(self, sk_mode=False):
        """
        :param sk_mode: If set to True, the values will be reshaped to work
         with scikit-learn.
        """
        if sk_mode:
            return [(key, value.reshape(1, -1)) for key, value in self.data.items()]
        else:
            return list(self.data.items())

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

        :return: a list of tuples containing the most similar words
         and their cosine similarities.
        """

        if not isinstance(coordinates, np.ndarray):
            coordinates = self.data[coordinates]

        if exclude is None:
            exclude = []

        similarity_dict = dict()

        for key, key_coordinates in self.items():
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

        if not isinstance(from1, np.ndarray): from1 = self.data[from1]
        if not isinstance(from2, np.ndarray): from2 = self.data[from2]
        if not isinstance(to1, np.ndarray): to1 = self.data[to1]

        return self.most_similar(to1 - from1 + from2, best, exclude, similarity)

    def doesnt_match(self, keys: list) -> str:
        """
        Search the intruder in a dictionary key list.
        """

        average_vector = np.mean([self.data[key] for key in keys], axis=0)

        max_distance = 0
        max_distance_key = None
        for key in keys:
            vector = self.data[key]
            vector_distance = scipy.spatial.distance.cosine(average_vector,
                                                            vector)
            if vector_distance > max_distance:
                max_distance = vector_distance
                max_distance_key = key

        return max_distance_key


def word2vec_vectorizer(
        corpus: Union[Tokens, Sequence[Tokens]],
        dimension: int,
        window: int = 4,
        iter: int = 1000,
        model=CBOW_MODEL
        ) -> VectorDict:
    """
    A vectorizer using the word2vec algorithm (working with Gensim).

    :param corpus: a list of Tokens objects or a Tokens object.

    :param dimension: the number of dimensions of word vectors.

    :param model: the model used by word2vec: CBOW (default) or Skip-Gram.

    :return: a VectorDict object containing the word vectors.
    """

    corpus, _ = handle_corpus(corpus)
    vector_dict = VectorDict()

    model = Word2Vec(corpus, size=dimension, window=window,
                     iter=iter, min_count=1, **model)

    for word in unique_words(sum(corpus, [])):
        vector_dict[word] = model.wv[word]

    return vector_dict


def word_vectorizer(
        corpus: Union[Tokens, Sequence[Tokens]],
        dimension: Optional[int] = None,
        window: int = 4,
        vocabulary: Optional[Sequence[str]] = None
        ) -> VectorDict:
    """
    A distributional word vectorizer constructing a co-occurrence matrix
    and applying a dimension reduction algorithm on it.

    :param corpus: a list of Tokens objects or a Tokens object.

    :param dimension: the number of dimensions of final word vectors.
     If None, the default value, no dimension reduction will be applied.

    :param vocabulary: the vocabulary used for vectorization in the form of a
     word list, or if None (default), the vocabulary would be extracted from the corpus.

    :return: a VectorDict object containing the word vectors.
    """

    def build_window(document: Sequence[str],index: int,window: int) -> List[str]:
        """
        Utility function to generate the context window around a word.

        :param document: the tokenized document (in a list form) containing
         the word to observe.

        :param index: the index of the word to observe in the document.

        :param window: the size of the context window.

        :return: the list of surrounding words in the context window.
        """

        left_selection = []
        right_selection = []
        for i, word in enumerate(reversed(document[:index])):
            if i < window:
                left_selection.append(word)

        for i, word in enumerate(document[index+1:]):
            if i < window:
                right_selection.append(word)

        return list(reversed(left_selection)) + right_selection

    def dimension_reduction(vector_dict: np.ndarray, to_dimension: int) -> np.ndarray:
        """
        Utility function to reduce the size of all vectors of a VectorDict.
        """

        raw_coordinates = np.array(vector_dict.values())

        SVD_model = TruncatedSVD(n_components=to_dimension)
        reduced_coordinates = SVD_model.fit_transform(raw_coordinates)

        reduced_vector_dict = VectorDict()
        for index, key in enumerate(vector_dict.keys()):
            reduced_vector_dict[key] = reduced_coordinates[index]

        return reduced_vector_dict

    corpus_tokens, corpus_texts = handle_corpus(corpus)
    vector_dict = VectorDict()

    if vocabulary is None:
        vocabulary = unique_words(sum(corpus_tokens, []))

    vector_dict.unique_words = vocabulary

    for word in vocabulary:
        word_vectors = np.zeros(len(vocabulary))
        word_count = 0

        for document in corpus_tokens:
            # lists all word indexes in the document.
            document_word_indexes = [index for index, doc_word in enumerate(document) if doc_word == word]
            word_count += len(document_word_indexes)

            for index in document_word_indexes:
                # Creates a vector for each index and adds it to the total word vector.
                selection = build_window(document, index, window)

                word_vectors = np.add(word_vectors,
                                      np.array([selection.count(unique_word) for unique_word in vocabulary]))

        # The final vector is the division of the total word vector by the number of times the word appears (= average vector).
        vector_dict[word] = word_vectors / word_count

    if dimension is not None:
        vector_dict = dimension_reduction(vector_dict, dimension)

    return vector_dict


def centroid_document_vectorizer(
        corpus: Union[Tokens, Sequence[Tokens]],
        word_vectors: VectorDict
        ) -> VectorDict:
    """
    A document vectorizer calculating the centroid
    of the vectors of the words it contains.

    :param corpus: a list of Tokens objects or a Tokens object.

    :param word_vectors: a VectorDict object containing the word vectors.

    :return: a VectorDict object containing the document vectors.
    """

    corpus_tokens, corpus_texts = handle_corpus(corpus)
    vector_dict = VectorDict()

    for tokens, text in zip(corpus_tokens, corpus_texts):
        vector_dict[text] = np.mean([word_vectors[token] for token in tokens if token in word_vectors],
                                    axis=0)

    return vector_dict


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
    vector_dict = VectorDict()

    if vocabulary is None:
        vocabulary = unique_words(sum(corpus_tokens, []))

    vector_dict.unique_words = vocabulary

    for text, tokens in zip(corpus_texts, corpus_tokens):
        vector_dict[text] = np.array([tokens.count(word) for word in vocabulary])

    return vector_dict
