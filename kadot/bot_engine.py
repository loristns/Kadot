from kadot.classifiers import NeuralClassifier
from kadot.models import CRFExtractor
from kadot.tokenizers import regex_tokenizer, Tokens
from kadot.utils import SavedObject
from kadot.vectorizers import VectorDict
from io import IOBase
import json
import logging
from typing import Any, Callable, Optional, Sequence, IO, Union

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
        self.last_intent = None
        self.event_flag = None
        self.intent_flag = None

    def __setitem__(self, key, value):
        if value:  # No empty data in the context
            self.data[key] = value
            self.data_track[key] = self.age

    def __delitem__(self, key):
        del self.data_track[key], self.data[key]

    def __getitem__(self, item):
        try:
            if self.data_track[item] + 3 < self.age:  # If data is too old
                del self.data[item], self.data_track[item]

            return self.data[item]
        except KeyError:
            return None

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
                 classifier=NeuralClassifier,
                 tokenizer: Callable[..., Tokens] = regex_tokenizer
                 ):
        """
        :param word_vectors: a VectorDict object containing the word vectors
         that will be used to train the classifier (optional).

        :param tokenizer: the word tokenizer to use.
        """

        self.classifier_fn = classifier
        self.classifier = None
        self.tokenizer = tokenizer
        self.word_vectors = word_vectors

        self.intents = {}
        self.entities = {}
        self.contexts = {}

    def add_entity(self, name: str, extractor: CRFExtractor):
        self.entities[name] = extractor

    def intent(self, samples: Union[Sequence[str], IO], entities: Sequence[str] = []):
        if isinstance(samples, IOBase):  # Handle JSON file as input
            samples = json.load(samples)

        def wrapper(intent_function):
            self.intents[intent_function.__name__] = Intent(
                name=intent_function.__name__,
                func=intent_function,
                entities=entities,
                samples=samples
            )

            return intent_function

        return wrapper

    def hidden_intent(self):
        return self.intent([], entities=[])

    def prompt(self,
               message: Any,
               key: str,
               callback: str,
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

            return self.intents[callback].run(raw, ctx)

        self.intents['_prompt'] = Intent(name='_prompt', func=_prompt)
        context.intent_flag = '_prompt'

        return message, context

    def option(self,
               message: Any,
               key: str,
               classifier,
               callback: str,
               context: Context):

        def _option(raw, ctx):
            """
            An intent to retrieve the user's input and put it's
            classification in the context.
            """

            best_class, best_proba = '', 0

            for e_class, e_proba in classifier.predict(raw).items():
                if e_proba >= best_proba:
                    best_class, best_proba = e_class, e_proba

            ctx[key] = best_class

            return self.intents[callback].run(raw, ctx)

        self.intents['_option'] = Intent(name='_option', func=_option)
        context.intent_flag = '_option'

        return message, context

    def _get_training_dataset(self):
        training_dataset = {}

        for intent in self.intents.values():
            for sample in intent.samples:
                training_dataset[sample] = intent.name

        return training_dataset

    def train(self):
        self.classifier = self.classifier_fn(
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

        output = []
        intent_output, context = self.intents[best_intent].run(text, context)
        output.append(intent_output)
        context.last_intent = best_intent
        context.step()

        while context.event_flag:
            intent = context.event_flag
            context.event_flag = None

            logger.info("Event flag for {}.".format(intent))
            event_output, context = self.intents[intent].run(text, context)
            context.last_intent = intent
            context.step()

            output.append(event_output)

        self.contexts[conversation] = context

        return output
