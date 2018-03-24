from kadot.tokenizers import corpus_tokenizer, LIGHT_DELIMITER_REGEX,\
    regex_tokenizer
from kadot.vectorizers import centroid_document_vectorizer, SKIP_GRAM_MODEL, word2vec_vectorizer
import re
from typing import Sequence
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


# TODO: Not ready
# Will use "Centroid-based Text Summarization through Compositionality of Word Embeddings" in the future
def summarizer(
        text: str,
        lenght: int,
        separator: str = '. ',
        vectorizer_config: dict = SUMMARIZER_WORD2VEC_CONFIGURATION
        ) -> str:
    """
    Summarize a text.

    :param text: input text to summarize.

    :param lenght: length (in number of sentences) of the summary.

    :param separator: separator that joins the sentences of
     the summary together.

    :param vectorizer_config: dictionary "kwargs" giving the custom parameters
     to `word2vec_vectorizer`.

    :return: a summary based on excerpts from the original text.
    """

    sentence_delimiter = re.compile("[.!?;\n]+")

    text_tokens = regex_tokenizer(text)

    word_vectors = word2vec_vectorizer(text_tokens, **vectorizer_config)
    text_vectors = centroid_document_vectorizer(text_tokens, word_vectors)

    sentences = regex_tokenizer(text, delimiter=sentence_delimiter).tokens
    sentences_tokens = corpus_tokenizer(sentences)
    sentences_tokens = [sentence for sentence in sentences_tokens if len(sentence.tokens)]  # Removes empty sentences.
    sentences_vectors = centroid_document_vectorizer(sentences_tokens, word_vectors)

    selected_sentences, _ = zip(*sentences_vectors.most_similar(text_vectors[0], lenght))
    selected_sentences = set(selected_sentences)

    # Make the final summary
    summary = ""
    for sentence in sentences:
        if sentence in selected_sentences:
            selected_sentences.remove(sentence)
            summary += sentence + separator

    return summary
