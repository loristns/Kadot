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

## [📎 Donald Trump Tweets Keywords Extractor](https://github.com/the-new-sky/Kadot/blob/master/examples/trump_keyword.py)
*⚠ This is an experiment*

In this example, we use WordVectorizer and the SemanticDocVectorizer together on [Donald Trump tweets archive](https://github.com/bpb27/trump-tweet-archive) to retrieve words which are most similar to each tweet : the keywords.

The force of this system is that keywords are not inevitably contained in the tweet.
The only issue is performance because run time is proportional to number of tweet in the training...


```
    Russians are playing @CNN and @NBCNews for such fools - funny to watch, they don't have a clue! @FoxNews totally gets it!
    [('russians', -1.0400478182904962), ('democrats', -1.0436341900777046), ...]
```


**More examples are coming**
