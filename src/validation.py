import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from feature_pipeline import FeaturePipeline
from sklearn.metrics import classification_report

from tokenizer import Tokenizer
from enhance_df import enhance_tokenization, enhance_bad_words, enhance_readability, enhance_pos_tag

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--classifier', type=str,
						default='nb', choices=['nb', 'lr', 'rf'],
						help='Classifier choices = nb: Naive Bayes, lr: Logistic Regression, rf: Random Forest. (default|recommended: %(default)s)')
	parser.add_argument('--alpha', nargs='+', type=float,
						default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
						help='Alpha values for classifier=nb. (default: %(default)s)')
	parser.add_argument('--C', nargs='+', type=float,
						default=[0.001, 0.1, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 7.5, 10.0],
						help='C values for classifier=lr. (default: %(default)s)')
	parser.add_argument('--test_size', type=float,
						default=0.10,
						help='Fraction of test size. (default: %(default)s)')
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
train_labels = np.array(train_df['class'].tolist())



train_X, valid_X, train_y, valid_y = train_test_split(train_df, train_labels, test_size=args.test_size, random_state=42)

p_features = FeaturePipeline().fit_transform(train_X)

if args.classifier == 'nb':
	for a in args.alpha:
		pipeline = Pipeline([
			('all_features', p_features),
			('classifier', MultinomialNB(alpha=a))
		])
		pipeline.fit(train_X, train_y)

		preds = pipeline.predict(valid_X)
		print("alpha={0}, accuracy={1}".format(a, np.mean(preds == valid_y)))
		print(classification_report(valid_y, preds, target_names=['class 0','class 1','class 2']))

elif args.classifier == 'lr':
	for c in args.C:
		pipeline = Pipeline([
			('all_features', p_features),
			('classifier', LogisticRegression(C=c, class_weight='balanced', random_state=42, max_iter=args.max_iter, solver=args.solver,
								multi_class='multinomial'))
		])

		pipeline.fit(train_X, train_y)

		preds = pipeline.predict(valid_X)
		print("C={0}, accuracy={1}".format(c, np.mean(preds == valid_y)))
		print(classification_report(valid_y, preds, target_names=['class 0','class 1','class 2']))

elif args.classifier == 'rf':
	pipeline = Pipeline([
			('all_features', p_features),
			('classifier', RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42))
		])

	pipeline.fit(train_X, train_y)

	preds = pipeline.predict(valid_X)
	print("accuracy={0}".format(np.mean(preds == valid_y)))

	print(classification_report(valid_y, preds, target_names=['class 0','class 1','class 2']))
