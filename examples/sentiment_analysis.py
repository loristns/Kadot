from kadot.models import BayesClassifier

# This is a tiny dataset collected on the title of IMDB reviews of "Star Wars: The Force Awakens"
train = {
    "Star Wars fans win again": 'positive',
    "Greatest movie of all time": 'positive',
    "Yes, it really is that good.": 'positive',
    "Beyond incredible!": 'positive',
    "This is the best Star Wars movie ever.": 'positive',
    "Far and way the greatest film of 2015.": 'positive',
    "The best movie of 2015!": 'positive',
    "Not the movie I paid to see": 'negative',
    "Unimaginative, cheap, no fantasy, lacked vision": 'negative',
    "Disappointment all around": 'negative',
    "Critical failure": 'negative',
    "Star Wars is dead!": 'negative',
    "Couldn't be more disappointed": 'negative',
    "Wow! I am very disappointed and upset!": 'negative'
}
test = [
    "Cheap failure",
    "By far the greatest movie I ever seen"
]

classifier = BayesClassifier(train)

for test_sample in test:

    best_class, best_value = '', 0
    for i_class, i_value in classifier.predict(test_sample).items():
        if i_value > best_value:
            best_value = i_value
            best_class = i_class

    print('"{}" is {}'.format(test_sample, best_class))