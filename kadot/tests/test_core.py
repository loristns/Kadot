import numpy as np
from kadot.core import VectorDictionary

to_vector_dict = {'a': np.array([0, 1]), 'b': np.array([2, 3]), 'c': np.array([4, 5]), 'd': np.array([0, 1])}
vector_dict = VectorDictionary(to_vector_dict, 2)


def test_similar():
    similarity_dict = dict(vector_dict.most_similar(vector_dict['a'], best=2))
    assert similarity_dict['a'] == similarity_dict['d'] == 1.0