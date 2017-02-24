from kadot.generators import MarkovGenerator
import json

with open('trump_tweets_2016.json', 'r') as tweet_file:  # Load the json tweet archive
    tweet_json = json.loads(tweet_file.read())

tweets = [tweet['text'] for tweet in tweet_json]  # Get a raw text list of tweets

generator = MarkovGenerator()
generator.fit(tweets)

for i in range(0, 100):
    print('---------------------------')
    print(generator.predict(25)[:140])  # Generate text with max 25 words and cut at 140 first character like a tweet.
