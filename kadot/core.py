from collections import OrderedDict, Counter
import numpy as np
from scipy import spatial
from sklearn.decomposition import TruncatedSVD


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
            for coordinates in vectors.values():
                if not isinstance(coordinates, np.ndarray):  # Check if pre-filled dictionary values are numpy arrays.
                    raise TypeError('`vectors` argument should contain numpy arrays values.')

                if not coordinates.size == dimension:  # Check if pre-filled dictionary respect `dimension` argument.
                    raise ValueError('`vectors` argument must contain values with a length that'
                                     ' should be equal to {0} not {1}'.format(self.dimension, coordinates.size))

            self.vectors = OrderedDict(vectors)
        else:
            self.vectors = OrderedDict()

    def __str__(self):
        return "VectorDictionary({})".format(self.items())

    def __repr__(self):
        return "VectorDictionary({})".format(repr(self.vectors))

    def __getitem__(self, key):
        return self.vectors[key]

    def __setitem__(self, key, coordinates):
        if len(coordinates) == self.dimension:  # Check if coordinates respect self.dimension.
            if isinstance(coordinates, np.ndarray):  # Check if coordinates are numpy arrays.

                self.vectors[key] = coordinates

            else:
                raise TypeError('`coordinates` argument should be a numpy array.')
        else:
            raise ValueError('`coordinates` argument length should be equal to {0} not {1}'
                             .format(self.dimension, len(coordinates)))

    def keys(self):
        return list(self.vectors.keys())

    def values(self):
        return list(self.vectors.values())

    def items(self):
        return list(self.vectors.items())

    def most_similar(self, coordinates, best=5):
        """
        Return the `best` most similar dict entries of `coordinates`.
        """

        similarity_dict = dict()

        for key, key_coordinates in self.items():
            similarity_dict[key] = 1 - spatial.distance.cityblock(key_coordinates, coordinates)  # 1 - distance = similarity

        return Counter(similarity_dict).most_common(best)

    def apply_translation(self, from1, to1, from2, best=5):
        return self.most_similar(to1 - from1 + from2, best)

    def reduce(self, to_dimension=2):
        """
        Perform a SVD reduction on the dict.

        :param to_dimension: New dict dimension
        :return: A new dict with values that respect `to_dimension` arg
        """
        raw_coordinates = np.array([coordinates for coordinates in self.values()])

        SVD_model = TruncatedSVD(n_components=to_dimension)
        reduced_coordinates = SVD_model.fit_transform(raw_coordinates)

        reduced_dict = VectorDictionary(dimension=to_dimension)
        for index, key in enumerate(self.keys()):
            reduced_dict[key] = reduced_coordinates[index]

        return reduced_dict