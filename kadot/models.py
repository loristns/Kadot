from kadot.preprocessing import tfidf
from kadot.tokenizers import corpus_tokenizer, regex_tokenizer, Tokens
from kadot.utils import SavedObject, unique_words
from kadot.vectorizers import centroid_document_vectorizer,\
    cosine_similarity, SKIP_GRAM_MODEL, VectorDict, word2vec_vectorizer
from collections import Counter
import logging
import re
from typing import Callable, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_WALNUT_CONFIGURATION = {
    'embedding_dimension': 20,
    'hidden_size': 10,
    'iter': 800,
    'learning_rate': 0.01,
    'mean_ratio': 0.75,
    'min_tolerance': 1e-04,
    'entity_size_range': (1, 10+1),
}

DEFAULT_SUMMARIZER_WORD2VEC_CONFIGURATION = {
    'dimension': 100,
    'window': 10,
    'iter': 1000,
    'model': SKIP_GRAM_MODEL
}


class EntityRecognizer(SavedObject):

    def __init__(self,
                 train: Dict[str, Tuple[str]],
                 tokenizer: Callable[..., Tokens] = regex_tokenizer,
                 configuration=DEFAULT_WALNUT_CONFIGURATION
                 ):
        """
        :param train: a dictionary that contains training samples as keys
         and the entities to extract as values.

        :param tokenizer: the word tokenizer to use.

        :param configuration: a Dict that contains the parameters about
         the model. It must contain these values :
          - `embedding_dimension`
          - `hidden_size`
          - `iter`
          - `learning_rate`
          - `mean_ratio`
          - `min_tolerance`
          - `entity_size_range`
        """
        import torch
        from torch import nn, optim

        class _WalnutNetwork(nn.Module):

            def __init__(self,
                         vocabulary_size,
                         embedding_dimension,
                         hidden_size
                         ):
                super(_WalnutNetwork, self).__init__()
                self.hidden_size = hidden_size

                self.embeddings = nn.Embedding(
                    vocabulary_size,
                    embedding_dimension
                )
                self.encoder = nn.LSTM(  # a LSTM layer to encode features
                    embedding_dimension,
                    self.hidden_size,
                    batch_first=True,
                )
                self.decoder = nn.Linear(self.hidden_size, 2)

            def forward(self, inputs):
                outputs = self.embeddings(inputs)
                outputs, _ = self.encoder(outputs)
                outputs = self.decoder(outputs)

                return outputs[0]

        # Initialize some variables
        self.configuration = configuration
        self.tokenizer = tokenizer

        train_samples, train_labels = zip(*train.items())
        train_tokens = corpus_tokenizer(train_samples, tokenizer=self.tokenizer)
        self.vocabulary = unique_words(sum([t.tokens for t in train_tokens], []) + ['<unknown>'])

        # Training
        self.model = _WalnutNetwork(
            len(self.vocabulary),
            self.configuration['embedding_dimension'],
            self.configuration['hidden_size']
        )
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configuration['learning_rate']
        )
        loss_fn = nn.CrossEntropyLoss()

        logger.info("Starting model training.")
        for epoch in range(self.configuration['iter']):
            epoch_losses = []

            for sentence, goal in zip(train_tokens, train_labels):
                x = torch.tensor(
                    self._tokens_to_indices(sentence)
                ).view(1, len(sentence))
                y = torch.tensor(
                    [1 if word in goal else 0 for word in sentence]
                )

                self.model.zero_grad()
                y_pred = self.model(x)

                loss = loss_fn(y_pred, y)
                epoch_losses.append(float(loss))

                loss.backward()
                optimizer.step()

            if epoch % round(self.configuration['iter'] / 10) == 0:
                mean_loss = torch.tensor(epoch_losses).mean()
                logger.info("Epoch {} - Loss : {}".format(epoch, float(mean_loss)))

        logger.info("Model training finished.")

    def predict(self, text: str) -> Tuple[Tuple[str], float]:
        import torch
        from torch.nn import functional as F

        text = self.tokenizer(text)

        x = torch.tensor(
            self._tokens_to_indices(text)
        ).view(1, len(text))
        prediction = F.softmax(self.model(x), dim=1)

        # Apply a special correction
        prediction -= self.configuration['min_tolerance']
        prediction -= self.configuration['mean_ratio'] * prediction.mean()
        prediction /= prediction.std()

        tokens_with_scores = dict(zip(text.tokens, prediction))
        grams_with_scores = [(('',), 0)]

        for n in range(*self.configuration['entity_size_range']):
            for gram in text.ngrams(n):
                grams_with_scores.append(
                    (gram, sum([float(tokens_with_scores[word][1]) for word in gram]))
                )

        return Counter(dict(grams_with_scores)).most_common(1)[0]

    def _tokens_to_indices(self, tokens):
        vector = []
        for token in tokens:
            if token in self.vocabulary:
                vector.append(self.vocabulary.index(token))
            else:
                vector.append(self.vocabulary.index('<unknown>'))

        return vector


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
