import sys
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from feature_pipeline import FeaturePipeline
from sklearn.metrics import classification_report

from enhance_df import enhance_tokenization, enhance_bad_words, enhance_readability, enhance_pos_tag

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
	parser.add_argument('--classifier', type=str,
						default='nb', choices=['nb', 'lr', 'rf'],
						help='Classifier choice mnb: Naive Bayes, lr: Logistic Regression, rf: Random Forest. (default|recommended: %(default)s)')
	parser.add_argument('--alpha', type=float,
						default=0.005,
						help='Alpha value for classifier=nb. (default: %(default)s)')
	parser.add_argument('--C', type=float,
						default=7.0,
						help='C value for classifier=lr. (default: %(default)s)')
	parser.add_argument('--max_iter', type=int,
						default=100,
						help='Maximum of iterations for classifier=lr. (default: %(default)s)')
	parser.add_argument('--solver', type=str,
						default='lbfgs', choices=['lbfgs', 'newton-cg', 'sag', 'saga'],
						help='Solver for classifier=lr (default: %(default)s)')
	parser.add_argument('--n_estimators', type=int,
						default=300,
						help='Number of estimators for classifier=rf. (default: %(default)s)')
	parser.add_argument('--max_depth', type=int,
						default=3,
						help='Max depth for classifier=rf. (default: %(default)s)')

	args = parser.parse_args()

	return args


args = parse_args()

train_df = pd.read_csv(sys.stdin, names=['blog', 'class'])

train_df = enhance_tokenization(train_df)
train_df = enhance_bad_words(train_df)
#train_df = enhance_readability(train_df)
#train_df = enhance_pos_tag(train_df)

train_X = train_df
train_y = np.array(train_df['class'].tolist())

p_features = FeaturePipeline().fit_transform(train_X)

classifier = None
if args.classifier == 'nb':
    classifier = ('classifier', MultinomialNB(alpha=args.alpha))
elif args.classifier == 'lr':
    classifier = ('classifier', LogisticRegression(C=args.C, class_weight='balanced', random_state=42, max_iter=args.max_iter, solver=args.solver,
	                        multi_class='multinomial'))
elif args.classifier == 'rf':
    classifier = ('classifier', RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42))

pipeline = Pipeline([
	('all_features', p_features),
	classifier
])

pipeline.fit(train_X, train_y)

for test_filename in args.test_files:
	test_df = pd.read_csv(os.path.join(args.test_data_dir, test_filename), names=['blog', 'class'])
	test_df = enhance_tokenization(test_df)
	test_df = enhance_bad_words(test_df)
	#test_df = enhance_readability(test_df)
	#test_df = enhance_pos_tag(test_df)
	test_labels = np.array(test_df['class'].tolist())
	preds_test = pipeline.predict(test_df)
	#print("accuracy={0}".format(np.mean(preds_test == test_labels)))
	#print(classification_report(test_labels, preds_test, target_names=['class 0','class 1','class 2']))
	print(preds_test)
	new_labels = pd.DataFrame({'class': preds_test})
	test_df['class'] = new_labels['class']
	test_df = test_df.drop(labels=['length', 'words', 'sentences', 'sentence_count', 'bad_word_count', 'has_bad_words'], axis=1)
	print(test_df.columns.tolist())
	test_df.to_csv('test_mystere_res.csv', index=False, header=False)

"""
import csv

with open("predictions.csv", 'w', newline='') as f:
	wr = csv.writer(f)
	wr.writerow(["Id", "class"])
	
	for i, p in enumerate(preds_test):
		wr.writerow((i,p))
"""