import re
import itertools

LIGHT_DELIMITER_REGEX = "[\r\t\n\v\f ]{1,}"
DELIMITER_REGEX = "[.,!?:;()[\]{}><+\-*/\\= \"'\r\t\n\v\f@^¨`~_|]{1,}"


class BaseTokenizer(object):

    def __init__(self):
        self.last_delimiters = []
        self.last_tokens = ""
        self.last_starts_with_delimiter = False  # This attribute say if the first element is a delimiter or a token.

    def tokenize(self, text):
        pass

    def rebuild_last(self, tokens):
        """
        Rebuild a list of tokens using the delimiters of the last tokenized text.
        """

        if tokens is None:
            tokens = self.last_delimiters

        # Zip the two list starting by a delimiter or a token
        if self.last_starts_with_delimiter:
            zipped = itertools.zip_longest(self.last_delimiters, tokens)
        else:
            zipped = itertools.zip_longest(tokens, self.last_delimiters)

        return ''.join([i for i in itertools.chain.from_iterable(zipped) if i])


class SpaceTokenizer(BaseTokenizer):
    """
    A simple tokenizer where word are separated by spaces.

    Examples
    --------
    >>> SpaceTokenizer().tokenize("This is a-text !")
    ['This', 'is', 'a-text', '!']
    """

    def tokenize(self, text):
        self.last_starts_with_delimiter = text.startswith(" ")
        self.last_delimiters = [" " for _ in range(text.count(" "))]
        self.last_tokens = text.split(" ")
        return self.last_tokens


class RegexTokenizer(BaseTokenizer):
    """
    A tokenizer where word are separated by special characters.

    Examples
    --------
    >>> RegexTokenizer().tokenize("This is a-text !")
    ['This', 'is', 'a', 'text']
    """

    def __init__(self, delimiter=DELIMITER_REGEX):
        BaseTokenizer.__init__(self)

        self.delimiter = re.compile(delimiter)

    def tokenize(self, text):
        self.last_starts_with_delimiter = self.delimiter.match(text) is not None
        self.last_delimiters = self.delimiter.findall(text)
        self.last_tokens = [word for word in self.delimiter.split(text) if word]

        return self.last_tokens


def tokenize(text, tokenizer=RegexTokenizer()):
    return tokenizer.tokenize(text)
