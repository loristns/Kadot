from kadot.classifiers import BayesClassifier

# This is a small improvised dataset :)
sentiment_texts = {
    "These pizzas are so good !": 'pos',
    "I'm allergic to pizzas": 'neg',
    "Flowers are a so good choice of gift": 'pos',
    "This gift is broken !": 'neg'
}

bayes = BayesClassifier()
bayes.fit(sentiment_texts)

print(bayes.predict(["I'm a bit allergic to these flowers !"]))