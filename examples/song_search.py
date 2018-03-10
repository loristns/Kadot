from kadot.fuzzy import extract
from kadot.tokenizers import corpus_tokenizer, regex_tokenizer

# Songs from Twenty One Pilots' Regional At Best album.
songs = corpus_tokenizer([
    "Guns For Hands",
    "Holding On To You",
    "Ode To Sleep",
    "Slowtown",
    "Car Radio",
    "Forest",
    "Glowing Eyes",
    "Kitchen Sink",
    "Anathema",
    "Lovely",
    "Ruby",
    "Trees",
    "Be Concerned",
    "Clear"
])

while True:
    search = input('Search a song : ')
    result = extract(regex_tokenizer(search), songs, 1)[0]

    print("I'm about {}% sure you meant {}".format(
        round(result[1]*100),
        result[0])
    )
