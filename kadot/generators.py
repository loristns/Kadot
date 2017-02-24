from .tokenizers import SafeCharTokenizer
from random import choice


class BaseGenerator(object):
    """
    A base class for building text generators.
    """

    def __init__(self, start_end_tokens=('START', 'END'), tokenizer=SafeCharTokenizer()):
        """
        :param start_end_tokens: A tuple that contain start and end tokens.
        """

        self.tokenizer = tokenizer
        self.start_token = start_end_tokens[0]
        self.end_token = start_end_tokens[-1]

        self.documents = []

    def fit(self, documents):
        """Prepare and save the documents."""

        if isinstance(documents, list):
            for doc in documents:
                self.documents.append([self.start_token] + self.tokenizer.tokenize(doc) + [self.end_token])
        else:
            self.documents.append([self.start_token] + self.tokenizer.tokenize(documents) + [self.end_token])

    def predict(self, max_word=30):
        pass


class MarkovGenerator(BaseGenerator):

    def fit(self, documents):
        BaseGenerator.fit(self, documents)
        self.markov_chain = dict()

        # Generate a basic markov chain
        for corpus in self.documents:
            for index, word in enumerate(corpus):
                if not word == self.end_token:
                    if word in self.markov_chain.keys():
                        self.markov_chain[word].append(corpus[index+1])
                    else:
                        self.markov_chain[word] = [corpus[index + 1]]
                else:
                    self.markov_chain[word] = []

    def predict(self, max_word=30, join_chain=""):
        next_word = self.start_token
        generated_suite = []

        for i in range(max_word):
            next_word = choice(self.markov_chain[next_word])
            if next_word == self.end_token:
                break
            else:
                generated_suite.append(next_word)

        return join_chain.join(generated_suite)
