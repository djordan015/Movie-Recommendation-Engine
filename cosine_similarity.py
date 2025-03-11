from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



text = ["London Paris London", "Paris Paris London"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)

[[2,1], [1, 2]]

print("Vocabulary:", vectorizer.vocabulary_)
print("Feature Names:", vectorizer.get_feature_names_out())
print("Bag of Words Representation:\n", X.toarray())