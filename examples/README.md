# Kadot Examples

Learn how to play with **Kadot** !

## [💻 Donald Trump Tweets Generator](https://github.com/the-new-sky/Kadot/blob/master/examples/trump_generator.py) 
In this example, we use MarkovGenerator (the Kadot implementation of [Markov Chains](https://en.wikipedia.org/wiki/Markov_chain)) on [Donald Trump tweets archive](https://github.com/bpb27/trump-tweet-archive) to imitate Donald Trump tweets.

As you can see, that's quite funny to use :

```
    ---------------------------
    I will be allowed to hide!
    ---------------------------
    I don't wait!"
    ---------------------------
    While I always complaining about me!
    ---------------------------
    Obama's happening!
    ---------------------------
```

## [❤ Sentiment analysis](https://github.com/the-new-sky/Kadot/blob/master/examples/sentiment_analysis.py)
In this example, we use BayesClassifier to classify if a text is positive or negative. (This example is running on a very small dataset with a limited vocabulary.)

This works well :

```
I'm a bit allergic to these flowers ! negative
```

## [🔑 Donald Trump Tweets Keywords Extractor](https://github.com/the-new-sky/Kadot/blob/master/examples/trump_keyword.py)
*⚠ This is an experiment*

In this example, we use WordVectorizer and the SemanticDocVectorizer together on [Donald Trump tweets archive](https://github.com/bpb27/trump-tweet-archive) to retrieve words which are most similar to each tweet : the keywords.

The force of this system is that keywords are not contained in the tweet.
The only issue is performance because run time is proportional to number of tweet in the training...


```
I would have done even better in the election, if that is possible, if the winner was based on popular vote - but would campaign differently
[('won', 0.47225554189527541), ('press', 0.45150754191118958), ('only', 0.40140639090240193), ('results', 0.3898159041831275)]

Campaigning to win the Electoral College is much more difficult &amp; sophisticated than the popular vote. Hillary focused on the wrong states!
[('questions', 0.62731961035017358), ('nytimes', 0.54830606716194752), ('election', 0.54493344074959438), ('world', 0.51620333141999886)]

Yes, it is true - Carlos Slim, the great businessman from Mexico, called me about getting together for a meeting. We met, HE IS A GREAT GUY!
[('honor', 0.49293058492558117), ('crack', 0.46473941234017313), ('friend', 0.37791344844294561)]
```

## [📎 Argument extractor](https://github.com/the-new-sky/Kadot/blob/master/examples/argument_extractor.py)
*⚠ This is an undocumented experiment*

In this example, we train an AdaBoost (Decision Trees) classifier to get the ability of extracting the name of a city in a weather request. It use the PositionalWordVectorizer which is built for this kind of purposes.

```
"the forecast for New-York"
['york', 'new-york']
```

**More examples are coming**
