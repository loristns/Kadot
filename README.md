![Kadot](https://github.com/the-new-sky/Kadot/raw/master/logo.png)

## A clean text analyser.

[![Code Health](https://landscape.io/github/the-new-sky/Kadot/master/landscape.svg?style=flat)](https://landscape.io/github/the-new-sky/Kadot/master)


Kadot just let you analyses a text easily.

```python
>>> hello_world = Text("Hello, I'm Kadot : the clean text analyser.")
>>> hello_world.tokens

['Hello', 'I', 'm', 'Kadot', 'the', 'clean', 'text', 'analyser']
```

## Include

- A Tokenizer
- A Word-Level (like word embedding) and a Text-Level vectorizer. 

That's all ! But **in a very near future :

- A powerful Text Classifier
- A Markov Text Generator 
- A modulable [Named Entity Recognizer](https://en.wikipedia.org/wiki/Named-entity_recognition)

The philosophy behind Kadot is clear, **never hardcode the language rules**: use unsupervised techniques for **support most languages**.
