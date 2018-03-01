import datetime
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
import re
from gensim.models import Word2Vec, KeyedVectors


print('Import word2vec')
filename = './model/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
wordBank = model.vocab.keys()
maxSeqLength = 80

while True:
	contents = []
	words = []
	seqLengthCount = []

	text = input("Text: ")
	if text:
		test_content = []
		test_content.append(text)

		for s in test_content:
			tokens = nltk.word_tokenize(s)
			seqLengthCount.append(len(tokens))

			for i in range(len(tokens)):
				tokens[i] = tokens[i].lower()
				words.append(tokens[i])

			contents.append(tokens)

		vectors = []
		wordsList = []
		vectorCount = 0
		temp = None
		for word in words:
			if word in wordBank:
				vectorCount += 1
				vector = model.get_vector(word)
				vectors.append(vector)
				wordsList.append(word)
				if temp is None:
					temp = np.array(vector)
				else:
					temp += np.array(vector)

		avgVector = temp / vectorCount
		vectors.append(avgVector)
		wordVectors = np.array(vectors)

		unknownVectorIndex = len(vectors) - 1
		numContents = 24
		ids = np.full((numContents, maxSeqLength), unknownVectorIndex, dtype='int32')
		contentCounter = 0
		for content in contents:
			indexCounter = 0
			for c in content:
				try:
					ids[contentCounter][indexCounter] = wordsList.index(c)
				except ValueError:
					ids[contentCounter][indexCounter] = unknownVectorIndex
				indexCounter += 1
			
			contentCounter += 1

		batchSize = 24
		lstmUnits = 64
		numClasses = 2
		iterations = 100000
		numDimensions = 300

		tf.reset_default_graph()
		labels = tf.placeholder(tf.float32, [batchSize, numClasses])
		input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
		data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
		data = tf.nn.embedding_lookup(wordVectors, input_data)

		lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
		lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
		value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

		weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
		bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
		value = tf.transpose(value, [1, 0, 2])
		last = tf.gather(value, int(value.get_shape()[0]) - 1)
		prediction = (tf.matmul(last, weight) + bias)

		correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver = tf.train.import_meta_graph('./models/pretrained_lstm.ckpt.meta')
			saver.restore(sess, './models/pretrained_lstm.ckpt')

			predictedSentiment = sess.run(prediction, {input_data: ids})[0]
			print(predictedSentiment)
			if predictedSentiment[0] > predictedSentiment[1]:
				print('Positive')
			else:
				print('Negative')


