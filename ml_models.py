'''
Runs simple machine learning models along with all their model metrics
'''


import pandas as pd
import os
import glob
import numpy as np

def balance_classes(a, b):
    extra = max(a,b) - min(a,b)
    remove = int(extra / 8)
    return remove

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

frames = list(non_frames.values())
X_non = np.concatenate((frames), axis=0)
y_non = np.repeat(0, X_non.shape[0])

frames = list(nas_frames.values())
X_nas = np.concatenate((frames), axis=0)
y_nas = np.repeat(1, X_nas.shape[0])

X = np.concatenate((X_non, X_nas), axis=0)
y = np.concatenate((y_non, y_nas), axis=0)

'''
Start running ml models
'''
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

sc = StandardScaler()
X_std = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.1, random_state=0)

skf = StratifiedKFold(n_splits=5, shuffle=True)

scores = {'Logistic': {},
          'SVM': {},
          'KNN': {},
          'Decision Trees': {},
          'Random Forest': {}}

print('Fitting Logistic...')
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
score = accuracy_score(y_true=y_test, y_pred=y_pred)
y_pred = lr.predict_proba(X_test)[:,1]
y_class = lr.predict(X_test)
scores['Logistic']['Accuracy'] = score
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)
f1 = f1_score(y_true=y_test, y_pred=y_class)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_class)
precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
scores['Logistic']['fpr'] = fpr
scores['Logistic']['tpr'] = tpr
scores['Logistic']['f1'] = f1
scores['Logistic']['precision'] = precision
scores['Logistic']['recall'] = recall
plt.plot(fpr, tpr)


print('Fitting SVM...')
svm = SVC(C=1.0, kernel='rbf', probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
score = accuracy_score(y_true=y_test, y_pred=y_pred)
y_pred = svm.predict_proba(X_test)[:,1]
y_class = svm.predict(X_test)
scores['SVM']['Accuracy'] = score
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)
f1 = f1_score(y_true=y_test, y_pred=y_class)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_class)
precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
scores['SVM']['fpr'] = fpr
scores['SVM']['tpr'] = tpr
scores['SVM']['f1'] = f1
scores['SVM']['precision'] = precision
scores['SVM']['recall'] = recall
plt.plot(fpr, tpr)


print('Fitting K Nearest Neighbor...')
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
score = accuracy_score(y_true=y_test, y_pred=y_pred)
y_pred = knn.predict_proba(X_test)[:,1]
y_class = knn.predict(X_test)
scores['KNN']['Accuracy'] = score
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)
f1 = f1_score(y_true=y_test, y_pred=y_class)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_class)
precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
scores['KNN']['fpr'] = fpr
scores['KNN']['tpr'] = tpr
scores['KNN']['f1'] = f1
scores['KNN']['precision'] = precision
scores['KNN']['recall'] = recall
plt.plot(fpr, tpr)


print('Fitting Decision Trees...')
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
score = accuracy_score(y_true=y_test, y_pred=y_pred)
y_pred = tree.predict_proba(X_test)[:,1]
y_class = tree.predict(X_test)
scores['Decision Trees']['Accuracy'] = score
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)
f1 = f1_score(y_true=y_test, y_pred=y_class)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_class)
precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
scores['Decision Trees']['fpr'] = fpr
scores['Decision Trees']['tpr'] = tpr
scores['Decision Trees']['f1'] = f1
scores['Decision Trees']['precision'] = precision
scores['Decision Trees']['recall'] = recall
plt.plot(fpr, tpr)


print('Fitting Random Forest Classifier...')
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
score = accuracy_score(y_true=y_test, y_pred=y_pred)
y_pred = rf.predict_proba(X_test)[:,1]
y_class = tree.predict(X_test)
scores['Random Forest']['Accuracy'] = score
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred)
f1 = f1_score(y_true=y_test, y_pred=y_class)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_class)
precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
scores['Random Forest']['fpr'] = fpr
scores['Random Forest']['tpr'] = tpr
scores['Random Forest']['f1'] = f1
scores['Random Forest']['precision'] = precision
scores['Random Forest']['recall'] = recall
plt.plot(fpr, tpr)
plt.show()
