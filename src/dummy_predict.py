import os
import sys
import pickle
import pandas as pd
from sklearn.dummy import DummyClassifier
from joblib import load

print(sys.argv)
model_name = sys.argv[1]
print(model_name)

df = pd.read_csv(sys.stdin, names=['blog', 'class'])

clf = load(model_name)

y = clf.predict(df['blog'])
base = os.path.split(model_name)[1]
out = f"./out/{base}.out"
pickle.dump([clf.classes_,y], open(out, 'wb'))
