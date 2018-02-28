'''
Creates model results and metrics for MLP model
'''

import os
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def balance_classes(a, b):
    extra = max(a,b) - min(a,b)
    remove = int(extra / 8)
    return remove

def test_model(model):
    for key, value in non_frames.items():
        value = mms.transform(value)
        y_pred = list(model.predict_classes(value, verbose=0))
        acc = y_pred.count(0)
        acc = acc / len(y_pred)
        print('Baby {} Accuracy: {}'.format(key, acc))
        
    for key, value in nas_frames.items():
        value = mms.transform(value)
        y_pred = list(model.predict_classes(value, verbose=0))
        acc = y_pred.count(1)
        acc = acc / len(y_pred)
        print('Baby {} Accuracy: {}'.format(key, acc))

path = os.getcwd()
os.chdir(path+'/non_decomp')
non_files = glob.glob('*.csv')
non_frames = {}

non_size = 0
for n in non_files:
    df = pd.read_csv(n)
    non_frames[n[:2]] = df
    non_size += df.shape[0]

os.chdir(path+'/nas_decomp')
nas_files = glob.glob('*.csv')
nas_frames = {}

nas_size = 0
for n in nas_files:
    df = pd.read_csv(n)
    nas_frames[n[:2]] = df
    nas_size += df.shape[0]

remove = balance_classes(non_size, nas_size)

frames = []
frames.extend(non_frames.values())
frames.extend(nas_frames.values())
X = pd.concat(frames, axis=0).values

#balance classes
for key, frame in non_frames.items():
    columns = frame.columns
    frame = pd.DataFrame(frame, columns = columns)
    non_frames[key] = frame

for key, frame in nas_frames.items():
    columns = frame.columns
    frame = frame.values[:-remove,:]
    frame = pd.DataFrame(frame, columns = columns)
    nas_frames[key] = frame

frames = []
frames.extend(non_frames.values())
frames.extend(nas_frames.values())
X = pd.concat(frames, axis=0).values

mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(X)
X = mms.transform(X)

frames = list(non_frames.values())
X_non = np.concatenate((frames), axis=0)
y_non = np.repeat(0, X_non.shape[0])

frames = list(nas_frames.values())
X_nas = np.concatenate((frames), axis=0)
y_nas = np.repeat(1, X_nas.shape[0])

X = np.concatenate((X_non, X_nas), axis=0)
X = mms.transform(X)
y = np.concatenate((y_non, y_nas), axis=0)

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
    
from sklearn.model_selection import StratifiedKFold


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scores = []

for train, test in kfold.split(X, y):
    model = Sequential()
    model.add(Dense(8, input_dim=X.shape[1], activation='relu', use_bias=True))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X[train], y[train], epochs=10, verbose=1)
    score = model.evaluate(X[test], y[test])
    scores.append(score[1]*100)
    
print(np.mean(scores), np.std(scores))

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

model = Sequential()
model.add(Dense(8, input_dim=X.shape[1], activation='relu', use_bias=True))
model.add(Dropout(rate=0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=1)
y_pred = model.predict_proba(X_test)
y_class = model.predict_classes(X_test)
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)
plt.plot(fpr, tpr)
plt.show()

f1 = f1_score(y_true=y_test, y_pred=y_class)

scores = {'MLP': {}}
scores['MLP']['fpr'] = fpr
scores['MLP']['tpr'] = tpr
scores['MLP']['f1'] = f1
'''
import pickle
os.chdir(path)
with open('roc_mlp.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''