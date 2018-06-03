from kadot.classifiers import NeuralClassifier
from kadot.models import CRFExtractor
from kadot.utils import SavedObject
from kadot.vectorizers import VectorDict
import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


class ConversationNode(SavedObject):

    def __init__(self, word_vectors: Optional[VectorDict] = None):
        """
        :param word_vectors: a VectorDict object containing the word vectors
         that will be used to train the classifier (optional).
        """

        self.word_vectors = word_vectors
        self.classifier = None
        self.entities = {}
        self.conversation_contexts = {}

        self.intent_functions = {}
        self.intent_samples = {}
        self.intent_entities = {}

    def intent(self, samples: Sequence[str]):

        def wrapper(intent_function):
            self.intent_functions[intent_function.__name__] = intent_function

            for sample in samples:
                self.intent_samples[sample.lower()] = intent_function.__name__

            return intent_function

        return wrapper

    def require_entity(self, name: str):

        def wrapper(intent_function):
            if intent_function.__name__ in self.intent_entities.keys():
                self.intent_entities[intent_function.__name__].append(name)
            else:
                self.intent_entities[intent_function.__name__] = [name]

            return intent_function

        return wrapper

    def add_entity(self, name: str, recognizer: CRFExtractor):
        self.entities[name] = recognizer

    def train(self):
        self.classifier = NeuralClassifier(
            self.intent_samples,
            word_vectors=self.word_vectors
        )

    def predict(self, text: str, conversation: Optional[Any] = None):
        best_intent, best_proba = '', 0

        for intent, proba in self.classifier.predict(text.lower()).items():
            logger.info("{}: {}".format(intent, proba))

            if proba >= best_proba:
                best_intent, best_proba = intent, proba

        # Retrieve entities
        extracted_entities = {}
        if best_intent in self.intent_entities.keys():
            for entity_name in self.intent_entities[best_intent]:
                extracted_entities[entity_name] = \
                    self.entities[entity_name].predict(text)

        context_entities = {}
        if conversation in self.conversation_contexts.keys():
            # Retrieve other entities from the context
            # (even if they are not required by the entities)
            context_entities = self.conversation_contexts[conversation]

        self.conversation_contexts[conversation] = extracted_entities

        # Extracted entities have priority over context entities
        entities = extracted_entities.copy()

        for entity_name, entity_value in context_entities.items():
            if entity_name in extracted_entities.keys():
                if not extracted_entities[entity_name][0] and entity_value[0]:
                    entities[entity_name] = entity_value
            else:
                entities[entity_name] = entity_value

        return self.intent_functions[best_intent](text, entities)
