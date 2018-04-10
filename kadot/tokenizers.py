from kadot.utils import SavedObject, unique_words
import itertools
import logging
import re
from typing import Callable, List, Optional, Pattern, Sequence, Union

logger = logging.getLogger(__name__)

LIGHT_DELIMITER_REGEX = re.compile("[\r\t\n\v\f ]+")
DELIMITER_REGEX = re.compile("[.,!?:;()[\]{}><+\-*/\\= \"'\r\t\n\v\f@^¨`~_|]+")


class Tokens(SavedObject):
    """
    An object representing a tokenized text.
    """

    def __init__(self,
                 text: str,
                 tokens: Sequence[str],
                 delimiters: Optional[Sequence[str]] = None,
                 starts_with_token: Optional[bool] = None,
                 exclude: Optional[Sequence[str]] = None
                 ):

        if exclude is None:
            exclude = []

        self.raw = text
        self.tokens = [token for token in tokens if token not in exclude]
        self.starts_with_token = starts_with_token
        self.delimiters = delimiters
        self.unique_words = unique_words(tokens)

    def __repr__(self):
        return "Tokens({})".format(repr(self.tokens))

    def __iter__(self):
        return self.tokens.__iter__()

    def rebuild(self, tokens: Union['Tokens', Sequence[str]]) -> str:
        """
        Allows you to reconstruct a modified raw text using
        the word delimiters of the original text.
        """

        if self.starts_with_token:
            zipped = itertools.zip_longest(tokens, self.delimiters)
        else:
            zipped = itertools.zip_longest(self.delimiters, tokens)

        return ''.join([i for i in itertools.chain.from_iterable(zipped) if i])

    def ngrams(self, n: int) -> list:
        """
        Returns n-grams of the text. Based on a code found on locallyoptimal.com
        """
        return list(zip(*[self.tokens[i:] for i in range(n)]))


def whitespace_tokenizer(
        text: str,
        lower: bool = False,
        exclude: Optional[Sequence[str]] = None
        ) -> Tokens:
    """
    Tokenize considering that words are separated by whitespaces characters.

    :param text: the text to tokenize.

    :param lower: if True, the text will be written in lowercase
     before it is tokenized.

    :param exclude: a list of words that should not be included
     after tokenization.

    :return: a Tokens object.

    Examples
    --------
    >>> whitespace_tokenizer("Let's try this example.")
    Tokens(["Let's", 'try', 'this', 'example.'])
    """

    if lower: text = text.lower()

    tokens = text.split(' ')
    delimiters = [' ' for _ in range(text.count(' '))]
    starts_with_token = not text.startswith(' ')

    return Tokens(text, tokens, delimiters, starts_with_token, exclude)


def regex_tokenizer(
        text: str,
        lower: bool = False,
        exclude: Optional[Sequence[str]] = None,
        delimiter: Pattern = DELIMITER_REGEX
        ) -> Tokens:
    """
    Tokenize using regular expressions.

    :param text: the text to tokenize.

    :param lower: if True, the text will be written in lowercase
     before it is tokenized.

    :param exclude: a list of words that should not be included
     after tokenization.

    :param delimiter: the regex defining what delimits words between them.

    :return: a Tokens object.

    Examples
    --------
    >>> regex_tokenizer("Let's try this example.")
    Tokens(['Let', 's', 'try', 'this', 'example'])
    """

    if lower: text = text.lower()

    tokens = [word for word in delimiter.split(text) if word]
    delimiters = delimiter.findall(text)
    starts_with_token = delimiter.match(text) is None

    return Tokens(text, tokens, delimiters, starts_with_token, exclude)


def ngram_tokenizer(
        text: str,
        n: int = 2,
        separator: str = '-',
        lower: bool = False,
        exclude: Optional[Sequence[str]] = None,
        tokenizer: Callable[..., Tokens] = regex_tokenizer
        ) -> Tokens:
    """
    A "meta" tokenizer that returns n-grams as tokens.

    :param text: the text to tokenize.

    :param n: the size of the gram.

    :param separator: the separator joining words together in a gram.

    :param lower: if True, the text will be written in lowercase
     before it is tokenized.

    :param exclude: a list of words and/or gram that should not be included
     after tokenization.

    :param tokenizer: the word tokenizer to use.

    :return: a Tokens object.

    Examples
    --------
    >>> ngram_tokenizer("This is another example.")
    Tokens(['This-is', 'is-another', 'another-example'])
    """

    tokenized_text = tokenizer(text, lower=lower, exclude=exclude)
    ngram = tokenized_text.ngrams(n)
    logger.info("The `Tokens.rebuild()` method will not be able"
                " to be called from the `Tokens` object"
                " returned by this tokenizer (ngram_tokenizer).")

    return Tokens(text, [separator.join(gram) for gram in ngram], exclude=exclude)


def corpus_tokenizer(
        corpus: List[str],
        lower: bool = False,
        exclude: Optional[Sequence[str]] = None,
        tokenizer: Callable[..., Tokens] = regex_tokenizer
        ) -> List[Tokens]:
    """
    Tokenize a whole list of documents (corpus) using the same tokenizer.

    Examples
    --------
    >>> corpus_tokenizer(['Hello bob !', 'Hi John !'])
    [Tokens(['Hello', 'bob']), Tokens(['Hi', 'John'])]
    """
    return [tokenizer(text, lower=lower, exclude=exclude) for text in corpus]
