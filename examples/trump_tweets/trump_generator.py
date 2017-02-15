from kadot.generators import MarkovGenerator
import json

with open('trump_tweets_2016.json', 'r') as tweet_file:
    tweet_json = json.loads(tweet_file.read())

tweets = [tweet['text'] for tweet in tweet_json]  # Get a raw text list of tweets

generator = MarkovGenerator()
generator.fit(tweets)

for i in range(0, 100):
    print('---------------------------\n{}'.format(generator.predict(25)[:140]))
