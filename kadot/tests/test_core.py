from kadot.core import VectorDictionary, VectorCoordinate

to_vector_dict = {'a': [0, 1], 'b': (2, 3), 'c': {4, 5}, 'd': [0, 1]}
vector_dict = VectorDictionary(to_vector_dict, 2)


def test_values():
    for value in vector_dict.values():
        assert isinstance(value, VectorCoordinate)


def test_coordinate_type():
    assert isinstance(vector_dict['c'].coordinates, list)


def test_similar():
    similarity_dict = dict(vector_dict.most_similar(vector_dict['a'], best=2))
    assert similarity_dict['a'] == similarity_dict['d'] == 1.0
