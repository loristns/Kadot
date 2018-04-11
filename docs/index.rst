.. Kadot documentation master file, created by
   sphinx-quickstart on Wed Apr 11 14:09:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Kadot : Unsupervised natural language processing.
=================================================

*⚠️ You are reading the documentation of Kadot 1.0 which is under development.*

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

**Kadot** is an open-source library to easily process text documents. It relies on vector representations of documents or words in order to solve NLP tasks such as **summarization**, **spellchecking** or **classification**.

::

   # How to get n-grams using kadot.
   >>> from kadot.tokenizers import regex_tokenizer
   >>> hello_tokens = regex_tokenizer("Kadot just lets you process a text easily.")
   >>> hello_tokens.ngrams(n=2)

   [('Kadot', 'just'), ('just', 'lets'), ('lets', 'you'), ('you', 'process'), ('process', 'a'), ('a', 'text'), ('text', 'easily')]


What's 🆕 in 1.0 ?
------------------

*⚠️ All these new features may not yet be available on Github.*

* **Vectorizers** : We are now offering Word2Vec, the state-of-the-art Fasttext and Doc2Vec algorithms using `Gensim <https://radimrehurek.com/gensim/>`_ 's powerful backend.
* **Performances** : Using a much more efficient algorithm, the new word vectorizer is up to 95% faster and sparse vectors now take up to 94% less memory.
* **Models** : Kadot now includes an *automatic text summarizer* and an *entity labeler* which can be useful in many projects.
* **Bot Engine** ?
* **Dependencies** 😞 : In order to guarantee good performance without reinventing the wheel, we are adding `Gensim <https://radimrehurek.com/gensim/>`_ and `Pytorch <http://pytorch.org/>`_ to our list of dependencies. Although installed by default, these libraries (with scikit-learn) will be optional and only Numpy and Scipy are strictly required to use Kadot.

⚖️ License
---------

Kadot is under `MIT license <https://github.com/the-new-sky/Kadot/blob/master/LICENSE.md>`_ .

*I am not a native English speaker, if you see any language mistakes in the documentation or in the code, please open an issue on Github.*
