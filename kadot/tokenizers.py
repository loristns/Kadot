WORD_TOKENS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n '


class BaseTokenizer(object):
    def tokenize(self, text):
        pass


class SpaceTokenizer(BaseTokenizer):
    """
    A simple tokenizer where word are separated by spaces.

    Examples
    --------
    >>> SpaceTokenizer().tokenize("This is a-text !")
    ['This', 'is', 'a-text', '!']
    """

    def tokenize(self, text):
        return text.split(" ")


class CharTokenizer(BaseTokenizer):
    """
    A tokenizer where word are separated by special characters.

    Examples
    --------
    >>> CharTokenizer().tokenize("This is a-text !")
    ['This', 'is', 'a', 'text']
    """

    def __init__(self, delimiters=WORD_TOKENS):
        self.delimiters = delimiters

    def tokenize(self, text):
        tokenized_text = []
        part_of_word = ""

        for character in text:
            if character in self.delimiters:
                if part_of_word != "":
                    tokenized_text.append(part_of_word)
                    part_of_word = ""
            else:
                part_of_word += character

        if part_of_word != "":
            tokenized_text.append(part_of_word)

        return tokenized_text


class SafeCharTokenizer(CharTokenizer):
    """
    Same as CharTokenizer, but save save punctuation. Used for generation tasks.

    Examples
    --------
    >>> SafeCharTokenizer().tokenize("This is a-text !")
    ['This ', 'is ', 'a-', 'text !']
    """

    def tokenize(self, text):
        split_regular_char = False
        tokenized_text = []
        part_of_word = ""

        for character in text:
            if split_regular_char:
                if part_of_word != "" and character not in self.delimiters:
                    tokenized_text.append(part_of_word)
                    part_of_word = ""
                    split_regular_char = False

            elif character in self.delimiters:
                split_regular_char = True

            part_of_word += character

        if part_of_word != "":
            tokenized_text.append(part_of_word)

        return tokenized_text