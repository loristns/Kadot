from kadot.vectorizers import BaseVectorizer
from kadot import Text


def test_unique_words():
    vectorizer = BaseVectorizer()
    vectorizer.fit("This is a test, a TEST")
    assert {'this', 'is', 'a', 'test'} == set(vectorizer.unique_words)  # Unique words don't take care of uppercase

def test_addition():
    hello_world = Text("Hello, I'm Kadot : the simplest text analyser of the world")
    hello_vectors = hello_world.vectorize(window=3, reduce_rate=2)
    vectors_result = hello_vectors.most_similar(hello_vectors['kadot'] + hello_vectors['analyser'], best=1)
    assert vectors_result[0][0] == 'text'
