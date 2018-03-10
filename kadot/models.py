from kadot.tokenizers import corpus_tokenizer, regex_tokenizer, Tokens
from kadot.utils import SavedObject
from kadot.vectorizers import count_document_vectorizer
from typing import Callable, Dict
from sklearn.naive_bayes import MultinomialNB


class ScikitClassifier(SavedObject):
    """
    A text classifier using scikit-learn.
    """

    def __init__(self,
                 train: Dict[str, str],
                 model=MultinomialNB(),
                 tokenizer: Callable[..., Tokens] = regex_tokenizer
                 ):

        """
        :param train: a dictionary that contains training samples as keys
         and their labels as values.

        :param model: the scikit-learn classifier to use.

        :param tokenizer: the word tokenizer to use.
        """

        self.model = model
        self.tokenizer = tokenizer

        train_samples, train_labels = zip(*train.items())
        train_tokens = corpus_tokenizer(train_samples, lower=True, tokenizer=self.tokenizer)
        train_vectors = count_document_vectorizer(train_tokens)

        self.vocabulary = train_vectors.unique_words

        self.model.fit(train_vectors.values(), train_labels)

    def predict(self, text: str) -> Dict[str, float]:

        message_tokens = self.tokenizer(text, lower=True)
        message_vector = count_document_vectorizer(message_tokens, self.vocabulary)
        prediction = self.model.predict_proba(message_vector[0].reshape(1, -1))[0]

        return dict(zip(self.model.classes_, prediction))

