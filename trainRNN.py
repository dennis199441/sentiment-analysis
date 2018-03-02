import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='tensorflow')
import datetime
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from gensim.models import KeyedVectors
from random import randint

df = pd.read_csv('./dataset/sum.csv', sep='\|\|', engine='python')
df = df.sort_values(by=['sentiment'])

# 0 22603 50156
contents = []
words = []
maxSeqLength = 150
seqLengthCount = []

for s in df['content']:
	tokens = nltk.word_tokenize(s)
	seqLengthCount.append(len(tokens))

	if len(tokens) > maxSeqLength:
		maxSeqLength = len(tokens)
	
	for i in range(len(tokens)):
		tokens[i] = tokens[i].lower()
		words.append(tokens[i])

	contents.append(tokens)

print('Prepare word2vec and wordBank')
filename = './model/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
wordBank = model.vocab.keys()
print('word2vec done!')
print('wordBank done!')
'''
Initialize model
'''
# model = Word2Vec(contents, min_count=1)
# wordVectors = model.wv


'''
Visualize by reducing dimention using PCA
'''
print('Prepare wordVectors')
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
wordVectors = np.array(vectors, dtype='float32')
print('wordVectors done!')

'''
Convert contents to ids matrix
'''
print('Prepare ids matrix')
unknownVectorIndex = len(vectors) - 1
numContents = len(contents)
ids = np.full((numContents, maxSeqLength), unknownVectorIndex, dtype='int32')
contentCounter = 0
for content in contents:
	print('----Processing content:', contentCounter, '/', numContents - 1, end='\r')
	indexCounter = 0
	for c in content:
		try:
			ids[contentCounter][indexCounter] = wordsList.index(c)
		except ValueError:
			ids[contentCounter][indexCounter] = unknownVectorIndex
		indexCounter += 1
	
	contentCounter += 1

print('ids matrix done!')


'''
RNN model 
'''
print('Prepare computational graph')
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
numDimensions = 300

'''
Helper functions for training network
'''
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(27603,50155)
            labels.append([1,0])
        else:
            num = randint(1,17603)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(17604,27602)
        if (num > 22603):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

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

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=tf.stop_gradient(labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)
print('Computational graph done!')

'''
Training
'''
print('Start training')
with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	tf.summary.scalar('Loss', loss)
	tf.summary.scalar('Accuracy', accuracy)
	merged = tf.summary.merge_all()
	logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
	writer = tf.summary.FileWriter(logdir, sess.graph)

	for i in range(iterations):
		print("Training iteration: %s" % i, end='\r')

		#Next Batch of reviews
		nextBatch, nextBatchLabels = getTrainBatch()
		sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

		#Write summary to Tensorboard
		if (i % 50 == 0):
		   	summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
		   	writer.add_summary(summary, i)
		
		#Save the network every 10,000 training iterations
		if (i % 10000 == 0 and i != 0):
			print('Save network:', i)
			save_path = saver.save(sess, "models/pretrained_lstm_large.ckpt")
			print("saved to %s" % save_path)

	save_path = saver.save(sess, "models/pretrained_lstm_large.ckpt")
	print("saved to %s" % save_path)

	writer.close()


	'''
	Testing
	'''
	testIterations = 10
	for i in range(testIterations):
		nextBatch, nextBatchLabels = getTestBatch()
		print('Accuracy for this batch:', (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)