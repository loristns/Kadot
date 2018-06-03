from kadot.models import CRFExtractor

# Find the city in a weather related query
train = {
    "What is the weather like in Paris ?": ('Paris',),
    "What kind of weather will it do in London ?": ('London',),
    "Give me the weather forecast for Berlin please.": ('Berlin',),
    "Tell me the forecast in New York !": ('New', 'York'),
    "Give me the weather in San Francisco...": ('San', 'Francisco'),
    "I want the forecast in Dublin.": ('Dublin',)
}
test = [
    "the forecast for Barcelona",
    "will it rain in Los Angeles ?",
    "Give me the weather !"
]

city_recognizer = CRFExtractor(train)

for test_sample in test:
    city = city_recognizer.predict(test_sample)

    print('"{}" -> {}'.format(test_sample, city))
