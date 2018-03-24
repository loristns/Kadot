import cProfile
from kadot.tokenizers import regex_tokenizer
from kadot.vectorizers import word_vectorizer

text = open('LouisVIII.txt').read()
tokens = regex_tokenizer(text)
cProfile.run("vec = word_vectorizer(tokens, 50)")

print(vec['Paris'].tolist())
print(vec['France'].tolist())