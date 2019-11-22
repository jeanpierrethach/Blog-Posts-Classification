import sys
import pickle
import pandas as pd 
from sklearn.dummy import DummyClassifier
from joblib import dump

df = pd.read_csv(sys.stdin, names=['blog', 'class'])

clf = DummyClassifier(strategy='most_frequent')
clf.fit(df['blog'], df['class'])

dump(clf, './models/dummy-most.clf')