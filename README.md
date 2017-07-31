![Kadot](https://github.com/the-new-sky/Kadot/raw/master/logo.png)

## Unsupervised natural language processing library.

[![Build Status](https://travis-ci.org/the-new-sky/Kadot.svg?branch=master)](https://travis-ci.org/the-new-sky/Kadot)  [![Code Health](https://landscape.io/github/the-new-sky/Kadot/master/landscape.svg?style=flat)](https://landscape.io/github/the-new-sky/Kadot/master) [![PyPI version](https://badge.fury.io/py/Kadot.svg)](https://badge.fury.io/py/Kadot) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/the-new-sky/Kadot/master/LICENSE.md) 


**Kadot** just lets you process a text easily.

```python
>>> hello_world = Text("Kadot just lets you process a text easily.")
>>> hello_world.ngrams(n=2)

[('Kadot', 'just'), ('just', 'lets'), ('lets', 'you'), ('you', 'process'), ('process', 'a'), ('a', 'text'), ('text', 'easily')]
```

## 🔋 What's included ?

Kadot includes **tokenizers**, text **generators**, **classifiers**, word-level and document-level **vectorizers** as well as a **spell checker**, a **fuzzy string matching** utility or a **stopwords detector**.

The philosophy of Kadot is *"never hardcode the language rules"* : use **unsupervised solutions** to support most languages. So it will never includes Treebank based algorithms (like a POS Tagger) : use [TextBlob](https://textblob.readthedocs.io/en/dev/) to do that.

## 🤔 How to use it ?
You can play with the TextBlob-like syntax :

```python
>>> from kadot import Text
>>> example_text = Text("This is a text sample !")
>>> example_text.words

['This', 'is', 'a', 'text', 'sample']

>>> example_text.ngrams(n=2)

[('This', 'is'), ('is', 'a'), ('a', 'text'), ('text', 'sample')]
```

Or you can use the words vectorizer to get words relations :

```python
>>> history_book = text_from_file('history_book.txt')
>>> vectors = history_book.vectorize(window=20, reduce_rate=300)
>>> vectors.apply_translation(vectors['man'], vectors['woman'], vectors['king'], best=1)

# 'man' is to 'woman' what 'king' is to...
[('queen', 0.98872148869)]
```

For more usages, check [examples](https://github.com/the-new-sky/Kadot/blob/master/examples).
An advanced documentation is coming.

## 🔨 Installation

Use the `pip` command that refair to you Python 3.x interpreter. 
In my case :
```
$ pip3 install kadot
```

It actually require the Python's standard library, Numpy, Scipy and Scikit-Learn.

## ⚖️ License
Kadot is under [MIT license](https://github.com/the-new-sky/Kadot/blob/master/LICENSE.md).

## 🚀 Contribute
Issues and pull requests are gratefully welcome. Come help us !

[![forthebadge](http://forthebadge.com/badges/built-with-love.svg)](http://forthebadge.com)
