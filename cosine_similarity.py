# test file to help understand how 
# cosine_similarity works 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


text = ["London Paris London", "Paris Paris London"]
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(text)

[[2,1], [1, 2]]

print("Vocabulary:", vectorizer.vocabulary_)
print("Feature Names:", vectorizer.get_feature_names_out())
print("Bag of Words Representation:\n", x.toarray())

print("cosine_similarity")
print(cosine_similarity(x))