'''
Models all possible sensor combinations using an LSTM RNN
'''

import pandas as pd
import os
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

class lstm:
    
    def __init__(self, sensors):
        self.sensors = sensors
        self.path = os.getcwd()
        self.import_data()
        self.build_3d_frames()
        self.kfold_cv()
        
    def import_data(self):
        os.chdir(self.path+'/non_decomp')
        non_files = glob.glob('*.csv')
        self.non_frames = {}
        
        non_size=0
        for n in non_files:
            df = pd.read_csv(n)
            for c in df.columns:
                check = any(x in c for x in self.sensors)
                if check == False:
                    df = df.drop(c, axis=1)
            non_size += df.shape[0]
            self.non_frames[n[:2]] = df
        
        os.chdir(self.path+'/nas_decomp')
        nas_files = glob.glob('*.csv')
        self.nas_frames = {}
        
        nas_size=0
        for n in nas_files:
            df = pd.read_csv(n)
            for c in df.columns:
                check = any(x in c for x in self.sensors)
                if check == False:
                    df = df.drop(c, axis=1)
            nas_size += df.shape[0]
            self.nas_frames[n[:2]] = df
        
        frames = []
        frames.extend(self.non_frames.values())
        frames.extend(self.nas_frames.values())
        X = pd.concat(frames, axis=0).values
        
        self.mms = MinMaxScaler(feature_range=(0, 1))
        self.mms.fit(X)
        X = self.mms.transform(X)
        self.remove = self.balance_classes(non_size, nas_size)
    
    def balance_classes(self, a, b):
        extra = max(a,b) - min(a,b)
        remove = int(extra / 8)
        return remove
        
    def build_3d_frames(self):
        
        for key, frame in self.non_frames.items():
            columns = frame.columns
            frame = self.mms.transform(frame.values)
            frame = pd.DataFrame(frame, columns = columns)
            X = self.reshape(frame)
            self.non_frames[key] = X
        
        for key, frame in self.nas_frames.items():
            columns = frame.columns
            frame = self.mms.transform(frame.values[:-self.remove,:])
            frame = pd.DataFrame(frame, columns = columns)
            X = self.reshape(frame)
            self.nas_frames[key] = X
        
        frames = list(self.non_frames.values())
        X_non = np.concatenate((frames), axis=0)
        y_non = np.repeat(0, X_non.shape[0])
        
        frames = list(self.nas_frames.values())
        X_nas = np.concatenate((frames), axis=0)
        y_nas = np.repeat(1, X_nas.shape[0])
        
        self.X = np.concatenate((X_non, X_nas), axis=0)
        y = np.concatenate((y_non, y_nas), axis=0)
        self.y_1d = y
        self.y = np_utils.to_categorical(y)
        
    def kfold_cv(self):
        
        print(self.X.shape)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        self.scores = []
        for k, (train, test) in enumerate(skf.split(self.X, self.y_1d)):
            print('Fold: {}'.format(k+1))
            self.X_train, self.X_test = self.X[train], self.X[test]
            self.y_train, self.y_test = self.y[train], self.y[test]
            self.scores.append(self.run_lstm())
            
        os.chdir(self.path)
        
    def reshape(self, df):
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

    def run_lstm(self):
        rnn = self.rnn()
        rnn.fit(self.X_train, self.y_train, epochs=5, verbose=0)
        loss, acc = rnn.evaluate(self.X_test, self.y_test, verbose=0)
        return acc
        
    def rnn(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(6, return_sequences=False),\
                                input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dropout(rate=0.3))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model

    def test_model(self, model):
        for key, value in self.non_frames.items():
            y_pred = model.predict(value)
            ind = [np.argmax(y, axis=0) for y in y_pred]
            acc = ind.count(0)
            acc = acc / len(ind)
            print('Baby {} Accuracy: {}'.format(key, acc))
            
        for key, value in self.nas_frames.items():
            y_pred = model.predict(value)
            ind = [np.argmax(y, axis=0) for y in y_pred]
            acc = ind.count(1)
            acc = acc / len(ind)
            print('Baby {} Accuracy: {}'.format(key, acc))
            
    def get_scores(self):
        return self.scores

sensors = ['LL','LA','C','RA','RL']
sensor_combinations = list(combinations(sensors, 2))
results = {}

import pickle

for s in sensor_combinations:
    print(s)
    model = lstm(s)
    results[s] = model.get_scores()

os.chdir(model.path+'/pickles')

with open('combination_2.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)