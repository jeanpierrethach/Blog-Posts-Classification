import sys
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tokenizer import Tokenizer

import tensorflow.keras.layers as L
from tensorflow.keras import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.backend import one_hot, clear_session
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer as KerasTokenizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--test_data_dir', type=str,
						default='../data',
						help='Directory path to the test datasets. (default: %(default)s)')
	parser.add_argument('--test_files', nargs='+', type=str,
						default=['test_split01.csv', 'test_split02.csv', 'test_split03.csv', 
						'test_split04.csv', 'test_split05.csv', 'test_split06.csv', 'test_split07.csv', 
						'test_split08.csv', 'test_split09.csv', 'test_split10.csv', 'test_split11.csv'],
						help='Names of the test files in test_data_dir. (default: %(default)s)')
	parser.add_argument('--vocab_size', type=int,
						default=10000,
						help='Vocabulary size for the embedding layer. (default: %(default)s)')
	parser.add_argument('--max_words', type=int,
						default=250,
						help='Maximum number of words per blog. (default: %(default)s)')
	parser.add_argument('--embedding_dim', type=int,
						default=100,
						help='Dimension of the embedding. (default: %(default)s)')
	parser.add_argument('--bilstm', action='store_true',
						help='Boolean indicating to use BiLSTM instead of LSTM')
	parser.add_argument('--units', type=int,
						default=32,
						help='Number of units in the LSTM layer. (default: %(default)s)')
	parser.add_argument('--spatial_dropout', type=float,
						default=0.4,
						help='Spatial dropout 1D. (default: %(default)s)')
	parser.add_argument('--dropout', type=float,
						default=0.4,
						help='Dropout of the LSTM layer. (default: %(default)s)')
	parser.add_argument('--recurrent_dropout', type=float,
						default=0.4,
						help='Recurrent dropout of the LSTM layer. (default: %(default)s)')
	parser.add_argument('--batch_size', type=int,
						default=1000,
						help='Batch size. (default: %(default)s)')
	parser.add_argument('--epochs', type=int,
						default=1,
						help='Number of epochs. (default: %(default)s)')

	args = parser.parse_args()

	return args


args = parse_args()


train_df = pd.read_csv(sys.stdin, names=['blog', 'class'])
train_labels = np.array(train_df['class'].tolist())

def vectorize_labels(labels):
	return np.array(labels)[:, np.newaxis]

class LSTMModel():
	def __init__(self, classes, vocabulary_size=args.vocab_size, max_words_per_comment=args.max_words, embedding_dim=args.embedding_dim):
		self.sequencer = KerasTokenizer(num_words=vocabulary_size,
								   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
								   lower=True)
		self.max_words_per_comment = max_words_per_comment
		self.vocabulary_size = vocabulary_size
		self.tokenizer = Tokenizer()
		self.classes = classes
		self.one_hot_encoder = OneHotEncoder()
		self.one_hot_encoder.fit(vectorize_labels(classes))        
		self.embedding_dim = embedding_dim
	
	def process_item(self, item):
		words = self.tokenizer.process_item(item)
		return ' '.join(words)
	
	def preprocess(self, data):
		X, y = data
		texts = [self.process_item(item) for item in X]
		self.sequencer.fit_on_texts(texts)
		X_seq = self.sequencer.texts_to_sequences(texts)
		X_seq = pad_sequences(X_seq, self.max_words_per_comment)
		y_onehot = self.one_hot_encoder.transform(vectorize_labels(y)).todense()
		return X_seq, y_onehot
	
	def build_model(self, input_shape):
		model = Sequential()
		model.add(L.Embedding(self.vocabulary_size, self.embedding_dim, input_length=input_shape))
		model.add(L.SpatialDropout1D(args.spatial_dropout))
		if args.bilstm:
			model.add(L.Bidirectional(L.LSTM(args.units, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout)))
		else:
   			model.add(L.LSTM(args.units, dropout=args.dropout, recurrent_dropout=args.recurrent_dropout))
		model.add(L.Dense(len(self.classes), activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		return model
	
	def prepare(self, train):
		self.train_data = self.preprocess(train)
		X, _ = self.train_data
		self.model = self.build_model(X.shape[1])
		
	def train(self, epochs=1):
		X, y = self.train_data
		history = self.model.fit(X, y, validation_data=None, epochs=epochs, batch_size=args.batch_size)
		print(history.history.keys())

		import matplotlib.pyplot as plt
		plt.plot(history.history['acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.savefig("accuracy_plot.png")

		plt.clf()
		plt.cla()
		plt.close()

		plt.plot(history.history['loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train'], loc='upper left')
		plt.savefig("loss_plot.png")

	def predict(self, test):
		X_seq, y_onehot = self.preprocess(test)
		return y_onehot[np.argmax(self.model.predict(X_seq), axis=1)]

	def evaluate(self, test):
		X_seq, y_onehot = self.preprocess(test)
		return self.model.evaluate(X_seq, y_onehot)

model = LSTMModel(list(set(train_labels)))
model.prepare((train_df['blog'], train_labels))
model.train(args.epochs)

for test_filename in args.test_files:
	test_df = pd.read_csv(os.path.join(args.test_data_dir, test_filename), names=['blog', 'class'])
	test_labels = np.array(test_df['class'].tolist())
	print(model.evaluate((test_df['blog'], test_labels)))