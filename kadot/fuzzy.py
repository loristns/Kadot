"""
Inspired by FuzzyWuzzy: https://github.com/seatgeek/fuzzywuzzy
"""
from kadot.tokenizers import regex_tokenizer, Tokens
from collections import Counter
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Sequence, Tuple, Union


def ratio(s1: Union[str, Tokens], s2: Union[str, Tokens]) -> float:
    """
    Compute the similarity ratio between two (tokenized or not) strings.

    Examples
    --------
    >>> ratio('I ate the apple', 'I ate the pear')
    0.759
    """

    if isinstance(s1, Tokens): s1 = s1.raw
    if isinstance(s2, Tokens): s2 = s2.raw

    return round(SequenceMatcher(a=s1, b=s2).ratio(), 3)


def token_ratio(s1: Tokens, s2: Tokens) -> float:
    """
    Compute the similarity ratio between two tokenized strings.

    Examples
    --------
    >>> token_ratio(regex_tokenizer('I ate the apple'), regex_tokenizer('the apple I ate'))
    0.5
    """
    return round(SequenceMatcher(a=s1.tokens, b=s2.tokens).ratio(), 3)


def vocabulary_ratio(s1: Tokens, s2: Tokens) -> float:
    """
    Compute the similarity ratio of the vocabulary of two tokenized strings.

    Examples
    --------
    >>> vocabulary_ratio(regex_tokenizer('I ate the apple'), regex_tokenizer('the apple I ate'))
    1.0
    """
    return round(SequenceMatcher(a=s1.unique_words, b=s2.unique_words).ratio(), 3)


def extract(
        query: Tokens,
        choices: Sequence[Tokens],
        best: Optional[int] = None,
        ratio_function: Callable[..., float] = ratio
        ) -> List[Tuple[str, int]]:
    """
    Find the `best` most similar `choices` to the `query`.

    :param query: a Tokens object.

    :param choices: a list of Tokens objects.

    :param best: the most similar number of choices to return.
     If None (default), the function will return all choices.

    :param ratio_function: a function calculating the similarity
     between two Tokens objects.

    :return: a list of tuple containing the plain text of the extracted choice
     and its similarity with the query.
    """

    similarity_dict = dict()

    for choice in choices:
        similarity_dict[choice.raw] = ratio_function(query, choice)

    return Counter(similarity_dict).most_common(best)