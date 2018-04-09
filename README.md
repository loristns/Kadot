![Kadot](https://github.com/the-new-sky/Kadot/raw/1.0dev/logo.png)
# Natural language processing using unsupervised vectors representation.
*⚠️ You are reading the README about Kadot 1.0 which is under development.*

**Kadot** is a high-level open-source library to easily process text documents. It relies on vector representations of documents or words in order to solve NLP tasks such as **summarization**, **spellchecking** or **classification**.

```python
# How to get n-grams using kadot.
>>> from kadot.tokenizers import regex_tokenizer
>>> hello_tokens = regex_tokenizer("Kadot just lets you process a text easily.")
>>> hello_tokens.ngrams(n=2)

[('Kadot', 'just'), ('just', 'lets'), ('lets', 'you'), ('you', 'process'), ('process', 'a'), ('a', 'text'), ('text', 'easily')]
```
## What's 🆕 in 1.0 ?
*⚠️ All these new features may not yet be available on Github.*

- **Vectorizers** : We are now offering Word2Vec, the state-of-the-art Fasttext and Doc2Vec algorithms using [Gensim](https://radimrehurek.com/gensim/)'s powerful backend.
- **Performances** : Using a much more efficient algorithm, the new word vectorizer is up to 95% faster and sparse vectors now take up to 94% less memory.
- **Models** : Kadot now includes an *automatic text summarizer* and an *entity labeler* which can be useful in many projects.
- **Bot Engine** ?
- **Dependencies** 😞 : In order to guarantee good performance without reinventing the wheel, we are adding [Gensim](https://radimrehurek.com/gensim/) and [Pytorch](http://pytorch.org/) to our list of dependencies. Although installed by default, these libraries (with scikit-learn) will be optional and only Numpy and Scipy are strictly required to use Kadot.

## ⚖️ License
Kadot is under [MIT license](https://github.com/the-new-sky/Kadot/blob/master/LICENSE.md).

## 🚀 Contribute
Issues and pull requests are gratefully welcome. Come help me !

*I am not a native English speaker, if you see any language mistakes in this README or in the code (docstrings included), please open an issue.*