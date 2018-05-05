import pickle


def unique_words(words):
    # TODO: add list of list/ list of Tokens objects gestion
    return sorted(set(words))


class SavedObject(object):
    """
    A class that can be saved in a file.
    """

    def save(self, filename):
        with open(filename, 'wb') as save_file:
            pickle.dump(self, save_file)


def load_object(filename):
    with open(filename, 'rb') as save_file:
        return pickle.load(save_file)
