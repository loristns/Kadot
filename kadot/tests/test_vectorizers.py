from unittest import TestCase

from kadot.tokenizers import regex_tokenizer
from kadot.vectorizers import word_vectorizer, count_document_vectorizer


class TestWordVectorizer(TestCase):

    def test_vector_order(self):
        vectors = word_vectorizer(regex_tokenizer("1 2 3 4 5"), window=1)
        self.assertEqual(vectors['1'][0], 0)
        self.assertEqual(vectors['1'][1], 1)


class TestCountDocumentVectorizer(TestCase):

    def test_vector(self):
        vectors = count_document_vectorizer(regex_tokenizer("1 2 3 4 5"),
                                            vocabulary="1 2 3".split())
        self.assertEqual(vectors["1 2 3 4 5"].shape, (3,))
