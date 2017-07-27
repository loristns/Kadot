from kadot import vectorizers, tokenizers

X = "Check the weather in London"
corpus = [
    "What the weather like in °argument° ?",
    "Give me the weather in °argument°",
    "Should I need an umbrella in °argument° ?",
    X
]

x_tokenized = tokenizers.RegexTokenizer().tokenize(X.lower())

word_vectorizer = vectorizers.PositionalWordVectorizer(window=len(x_tokenized)-1)
word_vecs = word_vectorizer.fit_transform(corpus)

arg_vectorizer = vectorizers.PositionalWordVectorizer(window=len(x_tokenized)-1)
arg_vectorizer.fit(X)
arg_vectorizer.synchronize(word_vectorizer)
arg_vecs = arg_vectorizer.transform()

print(X)
print([arg for arg in arg_vecs.most_similar(word_vecs['°argument°'], best=30) if arg[0] in x_tokenized])
