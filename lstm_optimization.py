'''
LSTM opimization is used to balance classes and reshape decomposed time series to be used in LSTM RNNs
class optimize performs 10-fold cross validation and can create data for many plots
 - A validation curve for hideen layer node size
 - Learning callbacks for loss and accuracies for different LSTM gradient descent methods
 - A learning curve for model performance as a function of data size
'''

import pandas as pd
import os
import glob
import numpy as np
import scipy as sp
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

def calc_stats(scores):
    stats = {}
    confidence = 0.95
    
    avg = np.average(scores)
    std_dev = np.std(scores)
    std_err = scipy.stats.sem(scores)
    bound = std_err * sp.stats.t._ppf((1 + confidence) / 2.0, len(scores))
    
    stats['avg_score'] = avg
    stats['ci_bound'] = bound
    stats['cv_scores'] = scores
    stats['std_dev'] = std_dev
    stats['std_err'] = std_err
    
    return stats

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
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split

#callback was changed
#example can be found at https://keras.io/callbacks/
class AccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class optimize:
    
    def __init__(self, X, y, sel):
        self.X = X
        self.y_split = y
        self.y = np_utils.to_categorical(y)
        switch = {'node':self.node, 'merge':self.merge, 'learn':self.learning}
        switch[sel]()
        
    def node(self):
        #lstm shape (samples, time steps, features)
        self.nodes = {}
        
        for n in range(1,16,1):
            scores = []
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
            split = kfold.split(self.X, self.y_split)
            for train, test in split:
                model = Sequential()
                model.add(Bidirectional(LSTM(n, return_sequences=False), input_shape=(X.shape[1], X.shape[2])))
                model.add(Dropout(rate=0.3))
                model.add(Dense(2, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
                model.fit(self.X[train], self.y[train], epochs=5, batch_size=100, verbose=1)
                score = model.evaluate(self.X[test], self.y[test])
                scores.append(score[1])
            stats = calc_stats(scores)
            self.nodes[n] = stats
            
    def merge(self):
        history = LossHistory()
        mode = ['sum','mul','ave','concat']
        self.loss = []
        for m in mode:
            model = Sequential()
            model.add(Bidirectional(LSTM(10, return_sequences=False),\
                                    input_shape=(X.shape[1], X.shape[2]), merge_mode=m))
            model.add(Dropout(rate=0.3))
            model.add(Dense(2, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
            model.fit(self.X, self.y, epochs=3, batch_size=1000, verbose=1, callbacks=[history])
            self.loss.append(history.losses)
        self.loss = np.array(self.loss).T
            
    def learning(self):
        #lstm shape (samples, time steps, features)
        self.learn = {}
        for n in range(1,10,1):
            train_acc = []
            test_acc = []
            stats = {}
            epochs=5
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_split, test_size=n/10, random_state=0)
            y_split = y_test
            y_test = np_utils.to_categorical(y_test)
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
            split = kfold.split(X_test, y_split)
            for train, test in split:
                model = Sequential()
                model.add(Bidirectional(LSTM(6, return_sequences=False), input_shape=(X.shape[1], X.shape[2])))
                model.add(Dropout(rate=0.3))
                model.add(Dense(2, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
                model.fit(X_test[train], y_test[train], epochs=epochs, verbose=1)
                score = model.evaluate(X_test[train], y_test[train])
                train_acc.append(score[1])
                score = model.evaluate(X_test[test], y_test[test], verbose=0)
                test_acc.append(score[1])
            stats['train_cv'] = train_acc
            stats['test_cv'] = test_acc
            self.learn[n/10] = stats

    def get_results(self):
        return self.learn
    
plt.style.use('ggplot')
#set plot settings
plt.rc('figure', figsize=(10,6))
#plt.rc('font', size=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

op = optimize(X, y, sel='learn')
hello = op.get_results()
'''
os.chdir(path+'/pickles')
import pickle
with open('learning_curve_4_epoch.pickle', 'wb') as handle:
    pickle.dump(hello, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''