from kadot.fuzzy import extract, ratio
from kadot.tokenizers import corpus_tokenizer, regex_tokenizer, Tokens
from kadot.utils import SavedObject, unique_words
from kadot.vectorizers import centroid_document_vectorizer, \
    count_document_vectorizer, DEFAULT_WORD2VEC_CONFIGURATION, VectorDict
import logging
from typing import Callable, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

NEURAL_CLASSIFIER_CONFIGURATION = {
    'hidden_size': 10,
    'iter': 800,
    'learning_rate': 0.01
}


class BayesClassifier(SavedObject):
    """
    A Naive Bayes classifier.
    """
    def __init__(self,
                 train: Dict[str, str],
                 tokenizer: Callable[..., Tokens] = regex_tokenizer
                 ):
        """
        :param train: a dictionary that contains training samples as keys
         and their classes as values.

        :param tokenizer: the word tokenizer to use.
        """

        self.tokenizer = tokenizer

        train_samples, train_labels = zip(*train.items())
        train_tokens = corpus_tokenizer(train_samples,
                                        tokenizer=self.tokenizer)

        # The concatenated tokenized training samples
        train_raw_tokens = sum([t.tokens for t in train_tokens], [])
        vectors = count_document_vectorizer(train_tokens)

        self.labels = unique_words(train_labels)  # List of possible classes
        self.vocabulary = unique_words(train_raw_tokens)
        self.n_labels = len(self.labels)
        self.n_words = len(self.vocabulary)

        # Estimate the probability that a document belongs to each class.
        self.label_proba = np.zeros(self.n_labels)

        for label_idx in range(self.n_labels):
            self.label_proba[label_idx] = \
                train_labels.count(self.labels[label_idx]) / len(train)

        # Estimate the probability that a word is contained in a document of each class.
        word_label_proba = np.zeros([self.n_words, self.n_labels])

        for doc_raw, doc_vec in vectors.g_items():
            word_label_proba[..., self.labels.index(train[doc_raw])] += doc_vec[0]

        word_label_proba /= np.sum(word_label_proba, axis=1).reshape([-1, 1])

        # Bayes' theorem
        self.feature_proba = {}

        for word_idx, word in enumerate(self.vocabulary):
            word_proba = train_raw_tokens.count(word) / len(train_raw_tokens)
            self.feature_proba[word] = (word_label_proba[word_idx] * word_proba) / self.label_proba

    def predict(self, text: str) -> Dict[str, float]:
        tokens = self.tokenizer(text)
        predicted_proba = np.zeros([1, self.n_labels])

        for token in tokens:
            if token in self.feature_proba:
                predicted_proba = np.append(
                    predicted_proba,
                    self.feature_proba[token].reshape([1, -1]),
                    axis=0
                )

        # Average the probabilities of each word
        predicted_proba = np.mean(predicted_proba, axis=0)

        class_prediction = {}
        for class_name, proba in zip(self.labels, predicted_proba):
            class_prediction[class_name] = float(proba)

        return class_prediction


class NeuralClassifier(SavedObject):
    """
    A word-embedding based shallow neural network classifier.
    """

    def __init__(self,
                 train: Dict[str, str],
                 word_vectors: Optional[VectorDict] = None,
                 configuration: dict = NEURAL_CLASSIFIER_CONFIGURATION,
                 tokenizer: Callable[..., Tokens] = regex_tokenizer,
                 vectorizer_config: dict = DEFAULT_WORD2VEC_CONFIGURATION
                 ):
        """
        :param train: a dictionary that contains training samples as keys
         and their classes as values.

        :param word_vectors: If provided, the VectorDict will be used as
         word vectors.

        :param configuration: a Dict that contains the parameters about
         the model. It must contain these values :
          - `hidden_size`
          - `iter`
          - `learning_rate`

        :param tokenizer: the word tokenizer to use.

        :param vectorizer_config: dictionary ("kwargs") giving the custom
         parameters to `word2vec_vectorizer`.
        """
        import torch
        from torch import nn, optim

        # Initialize some variables
        self.tokenizer = tokenizer
        self.word_vectors = word_vectors

        train_samples, train_labels = zip(*train.items())
        train_tokens = corpus_tokenizer(train_samples, tokenizer=self.tokenizer)

        if self.word_vectors is None:
            from kadot.vectorizers import word2vec_vectorizer
            self.word_vectors = word2vec_vectorizer(train_tokens,
                                                    **vectorizer_config)

        vectors = centroid_document_vectorizer(train_tokens, self.word_vectors)

        self.labels = unique_words(train_labels)
        n_labels = len(self.labels)

        # Initialize model
        self.model = nn.Sequential(
            nn.Linear(vectors.values().shape[1], configuration['hidden_size']),
            nn.Sigmoid(),
            nn.Linear(configuration['hidden_size'], n_labels),
            nn.Softmax(dim=1)
        )
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=configuration['learning_rate']
        )
        loss_fn = nn.NLLLoss()

        # Training
        logger.info("Starting model training.")
        for epoch in range(configuration['iter']):
            epoch_losses = []

            for sentence, label in zip(train_tokens, train_labels):
                x = torch.tensor([vectors[sentence.raw]]).float()
                y = torch.tensor([self.labels.index(label)])

                self.model.zero_grad()
                y_pred = self.model(x)

                loss = loss_fn(y_pred, y)
                epoch_losses.append(float(loss))

                loss.backward()
                optimizer.step()

            if epoch % round(configuration['iter'] / 10) == 0:
                mean_loss = torch.tensor(epoch_losses).mean()
                logger.info(
                    "Epoch {} - Loss : {}".format(epoch, float(mean_loss))
                )

        logger.info("Model training finished.")

    def predict(self, text: str) -> Dict[str, float]:
        import torch

        text = self.tokenizer(text)

        with torch.no_grad():
            x = torch.tensor([
                centroid_document_vectorizer(text, self.word_vectors)[text.raw]
            ]).float()

            prediction = self.model(x)

        class_prediction = {}
        for class_name, proba in zip(self.labels, prediction[0]):
            class_prediction[class_name] = float(proba)

        return class_prediction


class FuzzyClassifier(SavedObject):

    def __init__(self,
                 train: Dict[str, str],
                 ratio_function: Callable[..., float] = ratio,
                 tokenizer: Callable[..., Tokens] = regex_tokenizer
                 ):

        self.train_samples, self.train_labels = zip(*train.items())
        self.labels = unique_words(self.train_labels)

        self.ratio_function = ratio_function
        self.tokenizer = tokenizer

    def predict(self, text: str) -> Dict[str, float]:
        tokens = self.tokenizer(text)
        scores = extract(tokens, corpus_tokenizer(self.train_samples))

        class_prediction = {label: 0. for label in self.labels}

        for (sample, score) in scores:
            sample_label = self.train_labels[self.train_samples.index(sample)]
            if class_prediction[sample_label] < score:
                class_prediction[sample_label] = score

        return class_prediction
