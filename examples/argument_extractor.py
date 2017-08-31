from kadot.vectorizers import PositionalWordVectorizer
from kadot.tokenizers import tokenize, SpaceTokenizer
from sklearn.ensemble import AdaBoostClassifier


def dataset_generator(document):
    """
    This function generate texts from one trying to find a composed word.
    Examples
    --------
    >>> set(dataset_generator("I'm bob"))
    {'I m-bob', 'I-m bob', 'I m bob'}
    """

    tokenized_text = tokenize(document)

    # Generate unigrams, bigrams, ..., len(tokenized_text)-grams of the tokenized text
    ngrams = [list(zip(*[tokenized_text[i:] for i in range(n)])) for n in range(len(tokenized_text))]

    expended_texts = []

    for ngram in ngrams:
        for gram_index, gram in enumerate(ngram):
            tokens = []
            gram_placed = False

            for token_index, token in enumerate(tokenized_text):

                if gram_index <= token_index < gram_index + len(ngram) and not gram_placed:
                    gram_placed = True
                    tokens.append('-'.join(gram))
                else:
                    if token not in gram:
                        tokens.append(token)

            expended_texts.append(' '.join(tokens))

    return expended_texts


def get_unique_words(documents, tokenizer=SpaceTokenizer()):
    unique_words = set()
    for document in documents:
        for token in tokenize(document.lower(), tokenizer):
            unique_words.add(token)

    return list(unique_words)


def most_probable(probability_dict):
    bests = []
    highest_probability = -1

    for key, probability in probability_dict.items():
        if probability > highest_probability:
            highest_probability = probability
            bests = [key]

        elif probability == highest_probability:
            bests.append(key)

    return bests


if __name__ == '__main__':

    train_corpus = [
        "What is the weather like in °city°",
        "What kind of weather will it do in °city°",
        "Give me the weather forecast in °city°",
        "Tell me the forecast in °city°",
        "Give me the weather in °city°",
        "I want the forecast in °city°",

        "What is the weather like at °city°",
        "What kind of weather will it do at °city°",
        "Give me the weather forecast at °city°",
        "Tell me the forecast at °city°",
        "Give me the weather at °city°",
        "I want the forecast at °city°",

        "What is the weather like for °city°",
        "What kind of weather will it do for °city°",
        "Give me the weather forecast for °city°",
        "Tell me the forecast for °city°",
        "Give me the weather for °city°",
        "I want the forecast for °city°",

        "What is the weather like of °city°",
        "What kind of weather will it do of °city°",
        "Give me the weather forecast of °city°",
        "Tell me the forecast of °city°",
        "Give me the weather of °city°",
        "I want the forecast of °city°"
    ]

    query = input("Your query : ")
    adaboost_estimators = int(input("How many AdaBoost estimators ? (Works well with 100) >>> "))
    vectorizer_window = int(input("How large is the vectorizer window ? (Works well with 6) >>> "))

    extended_query = dataset_generator(query)
    query_unique_words = get_unique_words(extended_query)
    full_unique_words = list(set(get_unique_words(train_corpus) + query_unique_words))  # Get the features set

    # Training time !

    classifier = AdaBoostClassifier(n_estimators=adaboost_estimators)
    X = []
    Y = []

    for text in train_corpus:
        train_vectorizer = PositionalWordVectorizer(window=vectorizer_window, tokenizer=SpaceTokenizer())
        train_vectorizer.fit(dataset_generator(text))
        train_vectorizer.unique_words = full_unique_words

        train_vectors = train_vectorizer.transform()

        for word, vector in train_vectors.items():
            X.append(vector)
            Y.append(1 if word == '°city°' else 0)  # Classify if the vector is about a city or not

    classifier.fit(X, Y)

    # Prediction time !

    predict_vectorizer = PositionalWordVectorizer(window=vectorizer_window, tokenizer=SpaceTokenizer())
    predict_vectorizer.fit(extended_query)
    predict_vectorizer.unique_words = full_unique_words

    predict_vectors = predict_vectorizer.transform()

    proba_dict = {}

    for word, vector in predict_vectors.items():
        if word in query_unique_words:
            vector = vector.reshape(1, -1)
            proba_dict[word] = classifier.predict_proba(vector)[0][1]

    # Print the final result

    print('Training set :', train_corpus)
    print('Most probable arguments :', most_probable(proba_dict))
