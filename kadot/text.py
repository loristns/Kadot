from .tokenizers import CharTokenizer
from .vectorizers import WordVectorizer


class Text(object):
    """
    A TextBlob `Blob`-like class.

    Examples
    --------
    >>> Text("This is a-text !", tokenizer=CharTokenizer()).tokens
    ['This', 'is', 'a', 'text']
    >>> Text('This is another text !') + " " + "So fun"
    'This is another text ! So fun'
    """

    def __init__(self, text, tokenizer=CharTokenizer(), vectorizer=WordVectorizer()):

        self.raw_text = text
        self.words = self.tokens = tokenizer.tokenize(text)
        self.vectorizer = vectorizer

    def __str__(self):
        return self.raw_text

    def __repr__(self):
        return 'Text({})'.format(repr(self.raw_text))

    def __add__(self, other):
        return self.raw_text + other

    def vectorize(self, window=50, reduce=None):
        vectors = self.vectorizer.fit_transform(self.raw_text, window)

        if isinstance(reduce, int):
            return vectors.reduce(reduce)
        else:
            return vectors
