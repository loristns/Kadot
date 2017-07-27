import re

LIGHT_DELIMITER_REGEX = "[\r\t\n\v\f ]"
DELIMITER_REGEX = "[.,!?:;()[\]{}><+\-*/\\= \"'\r\t\n\v\f@^¨`~_|]"


class BaseTokenizer(object):
    def tokenize(self, text):
        pass


class SpaceTokenizer(BaseTokenizer):
    """
    A simple tokenizer where word are separated by spaces.

    Examples
    --------
    >>> SpaceTokenizer().tokenize("This is a-text !")
    ['This', 'is', 'a-text', '!']
    """

    def tokenize(self, text):
        return text.split(" ")


class RegexTokenizer(BaseTokenizer):
    """
    A tokenizer where word are separated by special characters.

    Examples
    --------
    >>> RegexTokenizer().tokenize("This is a-text !")
    ['This', 'is', 'a', 'text']
    """

    def __init__(self, delimiter=DELIMITER_REGEX):
        self.delimiter = delimiter

    def tokenize(self, text):
        return [word for word in re.split(self.delimiter, text) if word]
