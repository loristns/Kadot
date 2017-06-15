from kadot.vectorizers import WordVectorizer, SemanticDocVectorizer
import json

with open('trump_tweets_2016.json', 'r') as tweet_file:  # Load the json tweet archive
    tweet_json = json.loads(tweet_file.read())

tweets = [tweet['text'] for tweet in tweet_json]  # Get a raw text list of tweets
tweets = tweets[:300]  # Optional : Select 300 last tweets for quick results

tweet_vectorizer = SemanticDocVectorizer(window=2)
tweet_vectorizer.fit(tweets)
tweet_vectors = tweet_vectorizer.transform()

word_vectorizer = WordVectorizer(window=2)
word_vectorizer.fit(tweets)
word_vectors = word_vectorizer.transform()

for tweet in tweets:
    print("\n{}".format(tweet))
    print(word_vectors.most_similar(tweet_vectors[tweet]))