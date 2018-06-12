from kadot.classifiers import NeuralClassifier
from kadot.models import CRFExtractor
from kadot.tokenizers import regex_tokenizer, Tokens
from kadot.utils import SavedObject
from kadot.vectorizers import VectorDict
import logging
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)


class Context(object):
    """
    Keeps track of the data used in a conversation.
    """
    def __init__(self, name: str):
        self.name = name
        self.age = 0
        self.data = {}
        self.data_track = {}  # Indicate if data are expired
        self.intent_flag = None

    def __setitem__(self, key, value):
        if value or key not in self.data.keys():  # No empty data in the context
            self.data[key] = value
            self.data_track[key] = self.age

    def __delitem__(self, key):
        del self.data_track[key], self.data[key]

    def __getitem__(self, item):
        if self.data_track[item] + 2 < self.age:  # If data is too old
            self.data[item] = ''

        return self.data[item]

    def step(self):
        self.age += 1


class Intent(object):
    """
    A container for bot's intents.
    """

    def __init__(self,
                 name: str,
                 func: Callable[[str, Context], Any],
                 entities: Sequence[str] = [],
                 samples: Sequence[str] = []
                 ):
        self.name = name
        self.run = func
        self.entities = entities
        self.samples = samples


class Agent(SavedObject):

    def __init__(self,
                 word_vectors: Optional[VectorDict] = None,
                 tokenizer: Callable[..., Tokens] = regex_tokenizer
                 ):
        """
        :param word_vectors: a VectorDict object containing the word vectors
         that will be used to train the classifier (optional).

        :param tokenizer: the word tokenizer to use.
        """

        self.classifier = None
        self.tokenizer = tokenizer
        self.word_vectors = word_vectors

        self.intents = {}
        self.entities = {}
        self.contexts = {}

    def add_entity(self, name: str, extractor: CRFExtractor):
        self.entities[name] = extractor

    def intent(self, samples: Sequence[str], entities: Sequence[str] = []):

        def wrapper(intent_function):
            self.intents[intent_function.__name__] = Intent(
                name=intent_function.__name__,
                func=intent_function,
                entities=entities,
                samples=samples
            )

            return intent_function

        return wrapper

    def prompt(self,
               message: Any,
               key: str,
               callback: Callable[[str, Context], Any],
               context: Context):

        def _prompt(raw, ctx):
            """
            An intent to retrieve the user's input and put it in the context.
            """
            output = ''
            if key in self.entities.keys():
                # Try to use the entity extractor
                output = ' '.join(self.entities[key].predict(raw)[0])

            if output: ctx[key] = output
            else: ctx[key] = raw

            return callback(raw, ctx)

        self.intents['_prompt'] = Intent(name='_prompt', func=_prompt)
        context.intent_flag = '_prompt'

        return message, context

    def _get_training_dataset(self):
        training_dataset = {}

        for intent in self.intents.values():
            for sample in intent.samples:
                training_dataset[sample] = intent.name

        return training_dataset

    def train(self):
        self.classifier = NeuralClassifier(
            self._get_training_dataset(),
            word_vectors=self.word_vectors
        )

    def predict(self, text: str, conversation: Optional[Any] = None):
        if conversation in self.contexts.keys():
            context = self.contexts[conversation]
        else:
            context = Context(conversation)

        # Retrieve the intent
        best_intent, best_proba = '', 0

        if context.intent_flag is None:
            for intent, proba in self.classifier.predict(text.lower()).items():
                logger.info("{}: {}".format(intent, proba))

                if proba >= best_proba:
                    best_intent, best_proba = intent, proba
        else:  # Handle intent flag
            best_intent, best_proba = context.intent_flag, 1
            logger.info("Intent flag for {}.".format(best_intent))
            context.intent_flag = None

        # Retrieve entities
        for entity_name in self.intents[best_intent].entities:
            context[entity_name] = ' '.join(self.entities[entity_name].predict(text)[0])

        context.step()
        output, context = self.intents[best_intent].run(text, context)
        self.contexts[conversation] = context

        return output
