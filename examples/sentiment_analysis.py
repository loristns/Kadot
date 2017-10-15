from kadot import Text
from kadot.classifiers import ScikitClassifier
from sklearn.naive_bayes import MultinomialNB


# This is a tiny dataset collected on the title of IMDB reviews of "Star Wars: The Force Awakens"
reviews = {
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
    "Critical Failure": 'negative',
    "Star Wars is dead!": 'negative',
    "Couldn't be more disappointed": 'negative',
    "Wow! I am very disappointed and upset!": 'negative'
}

bayes = ScikitClassifier(MultinomialNB())
bayes.fit(reviews)

test = Text("I am never paying to see another Star Wars movie ever again.", classifier=bayes)

print('"{}" is {}'.format(test, test.classify()))
