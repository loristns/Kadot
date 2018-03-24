import json
from kadot.preprocessing import twitter_preprocess, summarizer
from kadot.tokenizers import corpus_tokenizer
from kadot.vectorizers import centroid_document_vectorizer, word2vec_vectorizer

print(summarizer("""Disponible dans certains pays, le HomePod d'Apple est passé sur le banc de test de nombreux médias anglo-saxons. Le verdict semble sans appel : c'est une enceinte merveilleuse mais un objet connecté au bas mot perfectible.

Avec le HomePod, Apple va tenter un nouveau pari : révolutionner, comme il aime le faire, le marché des enceintes connectées. En retard sur le secteur, la firme de Cupertino se doit de convaincre avec un produit à la pointe. Pas encore disponible chez nous, l’objet de toutes les curiosités a en revanche été lancé chez nos amis outre-Atlantique.

Et certains de nos confrères n’ont pas manqué de livrer leur verdict, arrivant plus ou moins tous aux mêmes conclusions : si le HomePod est une formidable enceinte, il reste un produit connecté très limité par rapport à la concurrence. Un peu comme si Apple n’avait fait que la moitié du chemin.
«  ll produit un son bien meilleur que les enceintes dans la même gamme de prix  », souligne Niley Patel de The Verge. On peut donc comprendre une chose : avec le HomePod, Apple a cherché avant tout à donner naissance à une enceinte de taille modeste ne lésinant pas sur la qualité sonore. Une prouesse qu’elle atteint grâce à une conception intelligente, offrant une spatialisation à nulle autre pareille qu’importe l’endroit où l’objet est placé.

Pas de Dolby machin chouette ou autre standard qui s’achète : juste la réalité d’un rendu qui en donne pour son argent. Plutôt que de chercher à ajouter des effets 3D marketing, le HomePod s’efforce d’éliminer tous les parasites liés à la configuration de la pièce grâce à des haut-parleurs virtuels et la puissance de la puce A8 (iPhone 6 et iPhone 6s). Pour WhatHifi, l’enceinte «  offre automatiquement le meilleur de votre musique, peu importe sa position  » et avec une calibration rapide et imperceptible, contrairement à la configuration pénible et aléatoire d’un Sonos.

Sans surprise, le HomePod est simple à utiliser et à configurer. Un savoir-faire qu’Apple perpétue d’un produit à l’autre et qui se vérifie une nouvelle fois ici. Sous couvert que vous adhériez à l’écosystème, la configuration se fait en quelques secondes depuis un iPhone. Fermé, le HomePod donne logiquement le meilleur de lui-même avec Apple Music : avec son enceinte connectée, Apple ne veut pas que l’on s’abonne à autre chose et cherche à vendre son service de streaming musical, le seul compatible avec les fonctions natives de l’enceinte.
Formidable enceinte, le HomePod pêche grandement du côté de son autre promesse, à savoir être un objet connecté, meilleur ami du quoditien. En tant qu’assistant vocal, Siri est à la traîne face à Google Home, Alexa et Cortana. Ainsi, selon Loup Ventures, le HomePod répond correctement à 52,3 % des questions, contre 81 % pour Google, 64 % pour Alexa et 57 % pour Cortana. Paradoxalement, il entend beaucoup mieux votre voix que les autres, même dans des environnements bruyants.
Toutefois, sur le HomePod, Siri est incapable de reconnaître précisément votre voix. En situation, tout le monde peut flouer l’enceinte pour accéder à des informations très personnelles. Cela donne également lieu à des situations assez étranges en termes d’usage : si mon iPhone est débloqué à côté de mon HomePod, il suffit de changer d’intonation pour que l’enceinte réponde à la place du téléphone qui, lui, est bien capable de différencier.

Autre inconvénient : s’il est possible de connecter son Apple TV via  AirPlay, il faudra le refaire à chaque fois que vous utilisez Siri sur l’enceinte ou jouez de la musique. Et vous pouvez oublier la reconnaissance vocale pour naviguer sur l’Apple TV, elle reste cantonnée à Apple Music et aux questions posées à Siri. Conclusion : le HomePod fait une bien mauvaise enceinte Bluetooth… même avec d’autres produits Apple.

En somme, le HomePod est mi-figue mi-raisin et plaira sans aucun doute aux fans d’Apple désireux d’avoir un meilleur rendu sonore chez eux. Ils devront juste se montrer indulgents face à certaines limites de la partie logiciel, limites que Apple sera obligé de lever à grand renfort de mise à jour. En tout cas, le HomePod ressemble à un produit à contre-courant : alors que les autres solutions sacrifient le son au profit du reste, il préfère s’adresser aux mélomanes… et aux plus tolérants d’entre nous côté finitions logicielles.
""", 5, '\n'))


with open('trump_tweets_2016.json') as tweet_file:
    tweet_data = json.load(tweet_file)

tweets = [twitter_preprocess(tweet['text']) for tweet in tweet_data][:100]

print(tweets)

tokenized_tweets = corpus_tokenizer(tweets, lower=True)

word_vectors = word2vec_vectorizer(tokenized_tweets, dimension=500, window=10)
tweet_vectors = centroid_document_vectorizer(tokenized_tweets, word_vectors)

print(tweet_vectors.most_similar(word_vectors['disrespect']))

print(word_vectors.doesnt_match('florida alabama pennsylvania russia'.split()))
print(word_vectors.most_similar('conspiracy'))