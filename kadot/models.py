from kadot.tokenizers import corpus_tokenizer, regex_tokenizer, Tokens
from kadot.utils import SavedObject, unique_words
from kadot.vectorizers import count_document_vectorizer
import logging
from typing import Callable, Dict, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SCAT_CONFIGURATION = {
    'embedding_dimension': 20,
    'hidden_size': 10,
    'iter': 800,
    'learning_rate': 0.01,
    'mean_ratio': 0.75,
    'min_tolerance': 1e-04,
    'entity_size_range': (1, 10+1),
}


class ScikitClassifier(SavedObject):
    """
    A text classifier using scikit-learn.
    """

    def __init__(self,
                 train: Dict[str, str],
                 model=None,
                 tokenizer: Callable[..., Tokens] = regex_tokenizer
                 ):
        """
        :param train: a dictionary that contains training samples as keys
         and their labels as values.

        :param model: the scikit-learn classifier to use.
         If None (default), MultinomialNB will be used.

        :param tokenizer: the word tokenizer to use.
        """
        from sklearn.naive_bayes import MultinomialNB

        if model is None:
            self.model = MultinomialNB()
        else:
            self.model = model

        self.tokenizer = tokenizer

        train_samples, train_labels = zip(*train.items())
        train_tokens = corpus_tokenizer(train_samples, tokenizer=self.tokenizer)
        train_vectors = count_document_vectorizer(train_tokens)

        self.vocabulary = unique_words(sum([t.tokens for t in train_tokens], []))
        self.model.fit(train_vectors.values(), train_labels)

    def predict(self, text: str) -> Dict[str, float]:

        message_tokens = self.tokenizer(text)
        message_vector = count_document_vectorizer(message_tokens, self.vocabulary)

        prediction = self.model.predict_proba(message_vector[0])[0]

        return dict(zip(self.model.classes_, prediction))


class EntityRecognizer(SavedObject):

    def __init__(self,
                 train: Dict[str, Tuple[str]],
                 tokenizer: Callable[..., Tokens] = regex_tokenizer,
                 configuration=DEFAULT_SCAT_CONFIGURATION
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
        from torch.autograd import Variable
        from torch import nn, optim

        # Initialize some variables
        self.configuration = configuration
        self.tokenizer = tokenizer

        train_samples, train_labels = zip(*train.items())
        train_tokens = corpus_tokenizer(train_samples, tokenizer=self.tokenizer)
        self.vocabulary = unique_words(sum([t.tokens for t in train_tokens], []) + ['<unknown>'])

        # Define the network
        class SCatNetwork(nn.Module):
            def __init__(self, vocabulary_size, embedding_dimension, hidden_size):
                super(SCatNetwork, self).__init__()

                self.embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
                self.encoder = nn.LSTM(  # a LSTM layer to encode features
                    embedding_dimension,
                    hidden_size,
                    batch_first=True,
                )
                self.decoder = nn.Linear(hidden_size, 2)

            def forward(self, inputs):
                hc = (
                    Variable(torch.ones(1, 1, 10)),
                    Variable(torch.ones(1, 1, 10))
                )

                outputs = self.embeddings(inputs)
                outputs, _ = self.encoder(outputs, hc)
                outputs = self.decoder(outputs)
                return outputs

        # Training
        self.model = SCatNetwork(
            len(self.vocabulary),
            self.configuration['embedding_dimension'],
            self.configuration['hidden_size']
        )
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configuration['learning_rate']
        )
        criterion = nn.CrossEntropyLoss()

        logger.info("Starting model training.")
        for epoch in range(self.configuration['iter']):
            epoch_losses = []

            for sentence, goal in zip(train_tokens, train_labels):

                sentence = sentence.tokens
                x = Variable(self._tokens_to_indices(sentence)).view(1, len(sentence))
                y = Variable(torch.LongTensor([1 if word in goal else 0 for word in sentence]))

                self.model.zero_grad()
                prediction = self.model(x)[0]

                loss = criterion(prediction, y)
                epoch_losses.append(float(loss))
                loss.backward()
                optimizer.step()

            if epoch % round(self.configuration['iter'] / 10) == 0:
                mean_loss = torch.FloatTensor(epoch_losses).mean()
                logger.info("Epoch {} - Loss : {}".format(epoch, float(mean_loss)))

        logger.info("Model training finished.")

    def predict(self, text: str) -> Tuple[Tuple[str], float]:
        from torch.autograd import Variable
        from torch.nn import functional as F

        text = self.tokenizer(text)

        x = Variable(self._tokens_to_indices(text.tokens)).view(1, len(text.tokens))
        prediction = F.softmax(self.model(x), dim=2)[0, :, 1].data

        # Apply a special correction
        prediction -= self.configuration['min_tolerance']
        prediction -= self.configuration['mean_ratio'] * prediction.mean()
        prediction /= prediction.std()

        # TODO: refactor to something more readable.
        tokens_with_scores = list(zip(text.tokens, prediction.tolist()))
        grams_with_scores = sum([list(zip(*[tokens_with_scores[i:] for i in range(n)])) for n in range(*self.configuration['entity_size_range'])], [])
        grams_with_scores.append([('', 0)])

        summed_gram_scores = [sum(list(zip(*gram))[1]) for gram in grams_with_scores]
        best_gram = list(zip(*grams_with_scores[summed_gram_scores.index(max(summed_gram_scores))]))

        return best_gram[0], sum(best_gram[1])

    def _tokens_to_indices(self, tokens):
        from torch import LongTensor

        vector = []
        for token in tokens:
            if token in self.vocabulary:
                vector.append(self.vocabulary.index(token))
            else:
                vector.append(self.vocabulary.index('<unknown>'))

        return LongTensor(vector)
