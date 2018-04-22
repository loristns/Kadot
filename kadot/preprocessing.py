from kadot.tokenizers import corpus_tokenizer, LIGHT_DELIMITER_REGEX,\
    regex_tokenizer, Tokens
from kadot.vectorizers import centroid_document_vectorizer, handle_corpus,\
    SKIP_GRAM_MODEL, word2vec_vectorizer
import math
import re
from typing import Dict, Sequence
from urllib.parse import urlparse

SUMMARIZER_WORD2VEC_CONFIGURATION = {
    'dimension': 300,
    'window': 20,
    'iter': 1000,
    'model': SKIP_GRAM_MODEL
}


def twitter_preprocess(
        text: str,
        no_rt: bool = True,
        no_mention: bool = False,
        no_hashtag: bool = False
        ) -> str:
    """
    Preprocessing function to remove retweets, mentions and/or hashtags from
    raw tweets.

    Examples
    --------
    >>> twitter_preprocess("RT @the_new_sky: Follow me !")
    'Follow me !'
    >>> twitter_preprocess("@the_new_sky please stop making ads for your #twitter account.", no_mention=True, no_hashtag=True)
    ' please stop making ads for your  account.'
    """

    if no_rt:
        text = re.sub(r"^RT @(?:[^ ]*) ", '', text)
    if no_mention:
        text = re.sub(r"@\w+", '', text)
    if no_hashtag:
        text = re.sub(r"#\w+", '', text)

    return text


def url_preprocess(text: str, exclude: Sequence[str] = ()) -> str:
    """
    Preprocessing function to remove some or all URLs from a text.

    :param text: input text to preprocess.

    :param exclude: a tuple or a list containing the domains to exclude,
     if empty (default) all URLs will be deleted.

    :return: the text without URLs.

    Examples
    --------
    >>> url_preprocess("Check this article : https://en.wikipedia.org/wiki/Python_(programming_language)")
    'Check this article : '
    >>> url_preprocess("https://www.bing.com/search?q=python or https://www.google.fr/search?q=python", exclude=('www.bing.com'))
    ' or https://www.google.fr/search?q=python'
    """

    tokenized_text = regex_tokenizer(text, delimiter=LIGHT_DELIMITER_REGEX)

    updated_tokens = []
    for token in tokenized_text:
        url_domain = urlparse(token).netloc

        if url_domain == '' or exclude and url_domain not in exclude:
            updated_tokens.append(token)
        else:
            updated_tokens.append('')

    return tokenized_text.rebuild(updated_tokens)


def tfidf(document: Tokens, corpus: Sequence[Tokens]) -> Dict[str, float]:
    """
    Returns the TF-IDF score of words in a given tokenized document based
    on a given tokenized corpus.
    """

    corpus_tokens, _ = handle_corpus(corpus)
    score_dict = {}

    for word in document.unique_words:
        tf = document.tokens.count(word) / len(document.tokens)
        idf = math.log(len(corpus) / (1 + sum(1 for doc in corpus_tokens if word in doc)))

        score_dict[word] = tf*idf

    return score_dict
