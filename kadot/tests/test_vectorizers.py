from kadot.vectorizers import BaseVectorizer


def test_unique_words():
    vectorizer = BaseVectorizer(window=0)
    vectorizer.fit("This is a test, a TEST")
    assert {'this', 'is', 'a', 'test'} == set(vectorizer.unique_words)  # Unique words don't take care of uppercase