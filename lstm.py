'''
Runs the final LSTM model and evaluates on indivdual babies
'''

import pandas as pd
import os
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def balance_classes(a, b):
    extra = max(a,b) - min(a,b)
    remove = int(extra / 8)
    return remove

def reshape(df):
    step = 12
    features = []
    for c in df.columns:
        time_steps = []
        for i in range(step):
            series = df[c].shift(-i)
            time_steps.append(series[:-step])
        array_2d = pd.concat((time_steps), axis=1).values
        array_3d = array_2d.reshape(array_2d.shape[0], step, 1)
        features.append(array_3d)
    X = np.concatenate((features), axis=2)
    return X

def test_model(model):
    for key, value in non_frames.items():
        y_pred = model.predict(value)
        ind = [np.argmax(y, axis=0) for y in y_pred]
        acc = ind.count(0)
        acc = acc / len(ind)
        print('Baby {} Accuracy: {}'.format(key, acc))
        
    for key, value in nas_frames.items():
        y_pred = model.predict(value)
        ind = [np.argmax(y, axis=0) for y in y_pred]
        acc = ind.count(1)
        acc = acc / len(ind)
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

mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(X)
X = mms.transform(X)

#building 3d frames
for key, frame in non_frames.items():
    columns = frame.columns
    frame = mms.transform(frame.values)
    frame = pd.DataFrame(frame, columns = columns)
    X = reshape(frame)
    non_frames[key] = X

for key, frame in nas_frames.items():
    columns = frame.columns
    frame = mms.transform(frame.values[:-remove,:])
    frame = pd.DataFrame(frame, columns = columns)
    X = reshape(frame)
    nas_frames[key] = X

frames = list(non_frames.values())
X_non = np.concatenate((frames), axis=0)
y_non = np.repeat(0, X_non.shape[0])

frames = list(nas_frames.values())
X_nas = np.concatenate((frames), axis=0)
y_nas = np.repeat(1, X_nas.shape[0])

X = np.concatenate((X_non, X_nas), axis=0)
y = np.concatenate((y_non, y_nas), axis=0)

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, shuffle=True)
split = kfold.split(X, y)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

y = np_utils.to_categorical(y)

#lstm shape (samples, time steps, features)

model = Sequential()
model.add(Bidirectional(LSTM(6, return_sequences=False),\
                        input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(rate=0.3))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X, y, epochs=5, verbose=1)
score = model.evaluate(X, y)

test_model(model)


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

model = Sequential()
model.add(Bidirectional(LSTM(6, return_sequences=False),\
                        input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(rate=0.3))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=1)
acc = model.evaluate(X_test, y_test)
y_pred = model.predict_proba(X_test)
y_class = model.predict_classes(X_test)
fpr, tpr, _ = roc_curve(y_true=y_test[:,1], y_score=y_pred[:,1])
plt.plot(fpr, tpr)
plt.show()

f1 = f1_score(y_true=y_test[:,1], y_pred=y_class)
area = auc(x=fpr, y=tpr)

conf_mat = confusion_matrix(y_true=y_test[:,1], y_pred=y_class)
precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])


scores = {'RNN': {}}
scores['RNN']['fpr'] = fpr
scores['RNN']['tpr'] = tpr
scores['RNN']['f1'] = f1

import pickle

os.chdir(path+'/pickles')
with open('roc_rnn.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
