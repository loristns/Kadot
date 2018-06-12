from kadot.bot_engine import Agent
from kadot.models import CRFExtractor
import random

# Set up an extractor to retrieve place names in queries.
places_extractor = CRFExtractor({
    "For eg. at Suite 101, Johnshon Avenue in London.": ('Suite', '101', 'Johnshon', 'Avenue,', 'London'),
    "The soldiers clashed at Gettysburg.": ('Gettysburg',),
    "The treaty was signed at Versailles.": ('Versailles',),
    "Give me the forecast in New York !": ('New', 'York'),
    "I am on Brompton Road": ('Brompton', 'Road',),
    "The plane stops at Dallas on the way to San Francisco ": ('Dallas', 'San', 'Francisco'),
    "Yesterday, I went to Dublin.": ('Dublin',)
    },
    crf_filename='.crf_city_extractor_city'
)

bot = Agent()
bot.add_entity('place', places_extractor)


@bot.intent([
    'hi',
    'hello !',
    'goodbye',
    'bye bye',
    'ok',
    'thanks'
])
def smalltalk(raw, context):
    """
    An intent to answer to greetings and appreciations.
    Answer the same with an exclamation mark (!).
    """
    return raw + '!', context


@bot.intent([
    "What is the weather like in Paris ?",
    "What kind of weather will it do in London ?"
    "Give me the weather forecast for Berlin please.",
    "Tell me the forecast in New York !",
    "Give me the weather in San Francisco...",
    "I want the forecast in Dublin."
], entities=['place'])
def weather(raw, context):
    """
    Say the weather for a given place (random ;p).
    """

    if context['place']:
        answer = "In {}, it will be {}".format(context['place'], random.choice(['sunny', 'cloudy', 'rainy']))
        return answer, context
    else:
        return bot.prompt("In which city ?", key='place', callback=weather, context=context)


bot.train()

while True:
    print(bot.predict(text=input('> ')))
