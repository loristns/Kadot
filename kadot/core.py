from collections import OrderedDict, Counter
import operator
import json
import numpy as np
from scipy import spatial
from sklearn.decomposition import TruncatedSVD


class VectorCoordinate(object):

    def __init__(self, coordinates):
        self.coordinates = list(coordinates)

    def __str__(self):
        return self.coordinates.__str__()

    def __repr__(self):
        return "VectorCoordinate({})".format(repr(self.coordinates))

    def __len__(self):
        return len(self.coordinates)

    def __eq__(self, other):
        other = VectorCoordinate(other)
        return self.coordinates == other.coordinates

    def __add__(self, other):
        other = VectorCoordinate(other)
        return VectorCoordinate(map(operator.add, self.coordinates, other.coordinates))

    def __sub__(self, other):
        other = VectorCoordinate(other)
        return VectorCoordinate(map(operator.sub, self.coordinates, other.coordinates))

    def __iter__(self):
        for dimensions in self.coordinates:
            yield dimensions

    def compare(self, other):
        """
        Return the Manhattan similarity between this coordinate and another one.
        """

        other = VectorCoordinate(other)
        return 1 - spatial.distance.cityblock(self.coordinates, other.coordinates)  # 1 - distance = similarity


class CoordinateJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, VectorCoordinate):
            return list(obj)
        else:
            return json.JSONEncoder.default(self, obj)


class VectorDictionary(object):
    """
    A dictionary that contain coordinates of a named document in vectorization (of words or text).
    """

    def __init__(self, vectors=None, dimension=3):
        """
        :param vectors: pre-filled dictionary
        :param dimension: length of all dictionary values
        """

        self.dimension = dimension

        if vectors is not None:
            for key, coordinates in vectors.items():
                if not isinstance(coordinates, VectorCoordinate):  # Convert all coordinate to VectorCoordinate objects.
                    vectors[key] = VectorCoordinate(coordinates)

                if not len(coordinates) == dimension:  # Check if pre-filled dictionary respect `dimension` argument.
                    raise ValueError('`vectors` argument must contain values with a length that'
                                     ' should be equal to {0} not {1}'.format(self.dimension, len(coordinates)))

            self.vectors = OrderedDict(vectors)
        else:
            self.vectors = OrderedDict()

    def __str__(self):
        return self.vectors.__str__()

    def __repr__(self):
        return "VectorDictionary({})".format(repr(self.vectors))

    def __getitem__(self, key):
        return VectorCoordinate(self.vectors[key])

    def __setitem__(self, key, coordinates):
        if len(coordinates) == self.dimension:  # Check if coordinates respect self.dimension.
            self.vectors[key] = VectorCoordinate(coordinates)
        else:
            raise ValueError('`coordinates` argument length should be equal to {0} not {1}'
                             .format(self.dimension, len(coordinates)))

    def keys(self):
        return list(self.vectors.keys())

    def values(self):
        return list(self.vectors.values())

    def items(self):
        return list(self.vectors.items())

    def get_json(self):
        return json.dumps(self.vectors, cls=CoordinateJSONEncoder)

    def most_similar(self, coordinates, best=5):
        """
        Return the `best` most similar dict entries of `coordinates`.
        """

        coordinates = VectorCoordinate(coordinates)
        similarity_dict = dict()

        for key, key_coordinates in self.items():
            similarity_dict[key] = coordinates.compare(key_coordinates)

        return Counter(similarity_dict).most_common(best)

    def apply_vector(self, from1, from2, to, best=5):
        return self.most_similar(from2 - from1 + to, best)

    def reduce(self, to_dimension=2):
        """
        Perform a SVD reduction on the dict.

        :param to_dimension: New dict dimension
        :return: A new dict with values that respect `to_dimension` arg
        """

        raw_coordinates = np.array([list(coordinates) for coordinates in self.values()])

        SVD_model = TruncatedSVD(n_components=to_dimension)
        reduced_coordinates = SVD_model.fit_transform(raw_coordinates)

        reduced_dict = VectorDictionary(dimension=to_dimension)
        for index, key in enumerate(self.keys()):
            reduced_dict[key] = VectorCoordinate(reduced_coordinates[index])

        return reduced_dict