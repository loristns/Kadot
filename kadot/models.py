from kadot.preprocessing import tfidf
from kadot.tokenizers import corpus_tokenizer, regex_tokenizer, Tokens
from kadot.vectorizers import centroid_document_vectorizer, \
    cosine_similarity, DEFAULT_WORD2VEC_CONFIGURATION, SKIP_GRAM_MODEL, \
    VectorDict, word2vec_vectorizer
import logging
import re
from typing import Callable, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CRF_CONFIGURATION = {
    'c1': 0.01,
    'c2': 0.001,
    'max_iterations': 200,
    'feature.possible_transitions': True
}

DEFAULT_SUMMARIZER_WORD2VEC_CONFIGURATION = {
    'dimension': 100,
    'window': 10,
    'iter': 1000,
    'model': SKIP_GRAM_MODEL
}


class CRFExtractor(object):

    def __init__(self,
                 train: Dict[str, Tuple[str]],
                 word_vectors: Optional[VectorDict] = None,
                 tokenizer: Callable[..., Tokens] = regex_tokenizer,
                 crf_config: dict = DEFAULT_CRF_CONFIGURATION,
                 crf_filename: str = '.kadot_crf_extractor',
                 vectorizer_config: dict = DEFAULT_WORD2VEC_CONFIGURATION
                 ):
        """
        :param train: a dictionary that contains training samples as keys
         and the entities to extract as values.

        :param word_vectors: If provided, the VectorDict will be used as
         word vectors.

        :param tokenizer: the word tokenizer to use.

        :param vectorizer_config: dictionary ("kwargs") giving the custom
         parameters to `word2vec_vectorizer`
        """
        import pycrfsuite

        self.tokenizer = tokenizer
        self.word_vectors = word_vectors
        self.crf_filename = crf_filename

        train_samples, train_labels = zip(*train.items())
        train_tokens = corpus_tokenizer(train_samples, tokenizer=self.tokenizer)

        if self.word_vectors is None:
            from kadot.vectorizers import word2vec_vectorizer
            self.word_vectors = word2vec_vectorizer(train_tokens,
                                                    **vectorizer_config)

        trainer = pycrfsuite.Trainer(verbose=False)
        for tokens, label in zip(train_tokens, train_labels):
            x = self._get_features(tokens)
            y = [str(token in label) for token in tokens]

            trainer.append(x, y)

        trainer.set_params(crf_config)
        trainer.train(self.crf_filename)

    def predict(self, text: str) -> Tuple[Tuple[str], float]:
        import pycrfsuite

        tagger = pycrfsuite.Tagger()
        tagger.open(self.crf_filename)

        text = self.tokenizer(text)
        features = self._get_features(text)
        prediction = tagger.tag(features)

        return (tuple([w for w, p in zip(text, prediction) if p == 'True']),
                tagger.probability(prediction))

    def _get_features(self, tokens):
        features = []

        for idx, token in enumerate(tokens):
            word_feature = [
                'bias',
                'word.lower=' + token.lower(),
                'word.last3=' + token[-3:],
                'word.last2=' + token[-2:],
                'word.first3=' + token[:3],
                'word.first2=' + token[:2],
                'word.is_upper=' + str(token.isupper()),
                'word.is_lower=' + str(token.islower()),
                'word.is_digit=' + str(token.isdigit()),
                'word.is_title=' + str(token.istitle())
            ]

            try:
                word_feature += ['word.vec{}={}'.format(*x) for x in enumerate(self.word_vectors[token])]
            except KeyError:
                pass

            if idx > 0:
                previous_token = tokens.tokens[idx-1]
                word_feature += [
                    '-1:word.lower=' + previous_token.lower(),
                    '-1:word.last3=' + previous_token[-3:],
                    '-1:word.last2=' + previous_token[-2:],
                    '-1:word.first3=' + previous_token[:3],
                    '-1:word.first2=' + previous_token[:2],
                    '-1:word.is_upper=' + str(previous_token.isupper()),
                    '-1:word.is_lower=' + str(previous_token.islower()),
                    '-1:word.is_digit=' + str(previous_token.isdigit()),
                    '-1:word.is_title=' + str(previous_token.istitle())
                ]

            if idx < len(tokens) - 1:
                next_token = tokens.tokens[idx+1]
                word_feature += [
                    '+1:word.lower=' + next_token.lower(),
                    '+1:word.last3=' + next_token[-3:],
                    '+1:word.last2=' + next_token[-2:],
                    '+1:word.first3=' + next_token[:3],
                    '+1:word.first2=' + next_token[:2],
                    '+1:word.is_upper=' + str(next_token.isupper()),
                    '+1:word.is_lower=' + str(next_token.islower()),
                    '+1:word.is_digit=' + str(next_token.isdigit()),
                    '+1:word.is_title=' + str(next_token.istitle())
                ]

            features.append(word_feature)

        return features


def summarizer(
        text: str,
        unrelated_corpus: Sequence[str],
        length: int,
        separator: str = '.',
        topic_threshold: float = 0.3,
        similarity_threshold: float = 0.95,
        vectorizer_config: dict = DEFAULT_SUMMARIZER_WORD2VEC_CONFIGURATION,
        word_vectors: Optional[VectorDict] = None
        ) -> str:
    """
    Summarize a text using "Centroid-based Text Summarization through
    Compositionality of Word Embeddings" by Gaetano Rossiello et al.

    :param text: input text to summarize.

    :param unrelated_corpus: a corpus of several and preferably unrelated
     documents with the text to be summarized. Used to determine tf-idf scores
     for each word and to improve the quality of word embeddings.

    :param length: length (in number of sentences) of the summary.

    :param separator: a string that joins the sentences of
     the summary together.

    :param topic_threshold: the minimum TF-IDF score so that a word is
     not identified as a stopword.

    :param similarity_threshold: the maximum cosine similarity value defining
     the redundancy of the selected sentences in the summary.

    :param vectorizer_config: dictionary ("kwargs") giving the custom
     parameters to `word2vec_vectorizer`.

    :param word_vectors: If provided, the VectorDict will be used as
     word vectors.

    :return: a summary based on excerpts from the original text.
    """

    text_tokens = regex_tokenizer(text, lower=True)
    sentences = regex_tokenizer(text, delimiter=re.compile("[.!?;\n]+"))

    unrelated_corpus_tokens = corpus_tokenizer(unrelated_corpus, lower=True)
    sentences_tokens = corpus_tokenizer(sentences, lower=True)
    # Removes empty sentences.
    sentences_tokens = [sentence for sentence in sentences_tokens if len(sentence.tokens)]

    if word_vectors is None:
        word_vectors = word2vec_vectorizer(text_tokens, **vectorizer_config)

    tfidf_scores = tfidf(text_tokens, unrelated_corpus_tokens)
    tokens_to_exclude = [token for token, score in tfidf_scores.items() if score < topic_threshold]

    # Log TF-IDF stats to help setting a better threshold.
    logger.info("TF-IDF - Min : {} - Max : {}"
                .format(
                    min(tfidf_scores.values()),
                    max(tfidf_scores.values())
                ))

    filtered_text_tokens = regex_tokenizer(text, lower=True, exclude=tokens_to_exclude)

    text_vector = centroid_document_vectorizer(filtered_text_tokens, word_vectors)[0]
    sentences_vectors = centroid_document_vectorizer(sentences_tokens, word_vectors)

    selected_sentences, _ = zip(*sentences_vectors.most_similar(text_vector, length))
    selected_sentences = set(selected_sentences)

    # Make the final summary
    summary = ""
    added_sentences = []

    for sentence in sentences:
        if sentence.lower() in selected_sentences:
            selected_sentences.remove(sentence.lower())

            is_invalid = False
            for As in added_sentences:
                is_invalid = is_invalid or cosine_similarity(sentences_vectors[As], sentences_vectors[sentence.lower()]) > similarity_threshold

            if not is_invalid:
                summary += sentence + separator
                added_sentences.append(sentence.lower())

    return summary
