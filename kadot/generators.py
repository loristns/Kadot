from .core import Fittable
from .tokenizers import RegexTokenizer, LIGHT_DELIMITER_REGEX
from random import choice


class BaseGenerator(Fittable):
    """
    A base class for building text generators.
    """

    def __init__(self, start_end_tokens=('START', 'END'), tokenizer=RegexTokenizer(LIGHT_DELIMITER_REGEX)):
        """
        :param start_end_tokens: A tuple that contain start and end tokens.
        """
        Fittable.__init__(self)

        self.tokenizer = tokenizer
        self.start_token = start_end_tokens[0]
        self.end_token = start_end_tokens[-1]

        self.tokenized_documents = []

    def fit(self, documents):
        """Prepare and save the documents."""
        Fittable.fit(self, documents)

        for doc in self.documents:
            self.tokenized_documents.append([self.start_token] + self.tokenizer.tokenize(doc) + [self.end_token])

    def predict(self, max_word=30):
        pass


class MarkovGenerator(BaseGenerator):

    def fit(self, documents):
        BaseGenerator.fit(self, documents)

        self.markov_chain = dict()

        # Generate a basic markov chain
        for corpus in self.tokenized_documents:
            for index, word in enumerate(corpus):
                if not word == self.end_token:
                    if word in self.markov_chain.keys():
                        self.markov_chain[word].append(corpus[index+1])
                    else:
                        self.markov_chain[word] = [corpus[index + 1]]
                else:
                    self.markov_chain[word] = []

    def predict(self, max_word=30, join_chain=" "):
        next_word = self.start_token
        generated_suite = []

        for _ in range(max_word):
            next_word = choice(self.markov_chain[next_word])
            if next_word == self.end_token:
                break
            else:
                generated_suite.append(next_word)

        return join_chain.join(generated_suite)
