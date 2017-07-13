from .tokenizers import CharTokenizer
from .vectorizers import WordVectorizer
from .generators import MarkovGenerator


class Text(object):
    """
    A TextBlob `Blob`-like class.

    :param classifier: Pass a trained classifier object.

    Examples
    --------
    >>> Text('This is a-text !', tokenizer=CharTokenizer()).tokens
    ['This', 'is', 'a', 'text']
    >>> Text('This is another text !') + " " + "So fun"
    'This is another text ! So fun'
    """

    def __init__(self, text, tokenizer=CharTokenizer(), vectorizer=WordVectorizer(), generator=MarkovGenerator(), classifier=None):

        self.raw_text = text
        self.words = self.tokens = tokenizer.tokenize(text)
        self.vectorizer = vectorizer
        self.generator = generator
        self.classifier = classifier

    def __str__(self):
        """
        Examples
        --------
        >>> str(Text('This is a-text !'))
        'This is a-text !'
        """
        return self.raw_text

    def __repr__(self):
        """
        Examples
        --------
        >>> Text('This is a-text !')
        Text('This is a-text !')
        """
        return 'Text({})'.format(repr(self.raw_text))

    def __add__(self, other):
        return self.raw_text + other

    def vectorize(self, window=50, reduce_rate=None):
        self.vectorizer.window = window
        vectors = self.vectorizer.fit_transform(self.raw_text)

        if isinstance(reduce_rate, int):
            return vectors.reduce(reduce_rate)
        else:
            return vectors

    def classify(self):
        if self.classifier is None:
            return None
        else:
            return self.classifier.predict([self.raw_text])[self.raw_text]

    def imitate(self, max_words=30):
        self.generator.fit([self.raw_text])
        return self.generator.predict(max_word=max_words)

    def ngrams(self, n=2):
        """
        Return n-grams of the text. Hazardously found in the locallyoptimal.com blog.

        :param n: length of grams

        Examples
        --------
        >>> Text('This is a-text !').ngrams(n=2)
        [('This', 'is'), ('is', 'a'), ('a', 'text')]
        """
        return list(zip(*[self.tokens[i:] for i in range(n)]))


def text_from_file(filename):
    """
    Generate a Text() object from the content of a file.

    :param filename: the filename of the file to read
    """
    with open(filename) as text_file:
        return Text(text_file.read())
