from kadot.generators import BaseGenerator, MarkovGenerator
from kadot.tokenizers import RegexTokenizer, LIGHT_DELIMITER_REGEX

def test_fit():
    generator = BaseGenerator()
    generator.fit(['hello world !'])

    assert generator.documents[0] == ['START', 'hello ', 'world !', 'END']


def test_maxlen():
    generator = MarkovGenerator()
    generator.fit('a b a b a a b b b a b')
    maxlen = 10

    assert len(RegexTokenizer(LIGHT_DELIMITER_REGEX).tokenize(generator.predict(maxlen))) <= maxlen
