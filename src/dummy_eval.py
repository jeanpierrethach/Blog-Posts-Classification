import os
import sys
import pickle
import pandas as pd
from sklearn.dummy import DummyClassifier
from joblib import load

from sklearn.metrics import classification_report

print(sys.argv)
predictions_file = sys.argv[1]
print(predictions_file)

df_test = pd.read_csv(sys.stdin, names=['blog', 'class'])

with open(f"./{predictions_file}",'rb') as f:
    preds = pickle.load(f)
    print(preds[1])
    print(classification_report(df_test['class'], preds[1]))

