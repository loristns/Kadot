"""
Freely inspired by FuzzyWuzzy: https://github.com/seatgeek/fuzzywuzzy
"""

from .tokenizers import RegexTokenizer
from collections import Counter
from difflib import SequenceMatcher


def ratio(string_a, string_b, ignore_case=False):
    """
    Compute the similarity ratio between two string.

    Examples
    --------
    >>> ratio('Hola', 'Hello')
    0.444
    """

    if ignore_case:
        string_a = string_a.lower()
        string_b = string_b.lower()

    return round(SequenceMatcher(a=string_a, b=string_b).ratio(), 3)


def vocabulary_ratio(string_a, string_b, ignore_case=False, tokenizer=RegexTokenizer()):
    """
    Get the vocabulary of two string and compute the ratio between them.

    Examples
    --------
    >>> vocabulary_ratio('This is a text !', 'This is a-text !')
    1.0
    """

    vocabulary_a = " ".join(sorted(set(tokenizer.tokenize(str(string_a)))))
    vocabulary_b = " ".join(sorted(set(tokenizer.tokenize(str(string_b)))))

    return ratio(vocabulary_a, vocabulary_b, ignore_case)


def most_similar(query, choices, best=None, ignore_case=False, ratio_function=ratio):
    """
    Return the `best` most similar `choices` with `query`. Useful for searching.

    Examples
    --------
    >>> most_similar("San Fr", ['New-York City', 'San Francisco', 'Los Angeles'], best=1)
    [('San Francisco', 0.632)]
    """

    similarity_dict = dict()

    for choice in choices:
        similarity_dict[choice] = ratio_function(query, choice, ignore_case)

    return Counter(similarity_dict).most_common(best)
