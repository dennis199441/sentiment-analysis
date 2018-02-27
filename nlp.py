import numpy as np
import pandas as pd
import nltk
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_csv('./dataset/test.csv', sep='\|\|', engine='python')
sentences = []
words = []

for s in df['content']:
	tokens = nltk.word_tokenize(s)
	sentences.append(tokens)
	for token in tokens:
		words.append(token)


filename = './model/GoogleNews-vectors-negative300.bin'
wordVectors = KeyedVectors.load_word2vec_format(filename, binary=True)
wordBank = wordVectors.vocab.keys()

'''
Initialize model
'''
# model = Word2Vec(sentences, min_count=1)
# wordVectors = model.wv


'''
Visualize by reducing dimention using PCA
'''
vectors = []
existWords = []
for word in words:
	if word in wordBank:
		vectors.append(wordVectors.get_vector(word))
		existWords.append(word)

print(len(vectors[0]))

pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(existWords):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()


'''
Logistic Regression 
'''
# X, y = df.content, df.sentiment
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
# count_vect = CountVectorizer(tokenizer=tokenizer.tokenize) 
# classifier = LogisticRegression()
# pipeline = Pipeline([
#         ('vectorizer', count_vect),
#         ('classifier', classifier)
#     ])

# pipeline.fit(X_train, y_train)
# score = pipeline.score(X_test, y_test)
# print('score:', score)

# print('predict:', pipeline.predict(['I like python though.']))
# print('predict_log_proba:', pipeline.predict_log_proba(['I like python though.']))
# print('predict_proba:', pipeline.predict_proba(['I like python though.']))