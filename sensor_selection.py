'''
Runs all 5 feature selection methods and graphs the results grouped for each sensor position
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


#set plot settings
plt.rc('figure', figsize=(20,8))
plt.rc('font', size=12)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=12)
plt.style.use('ggplot')

df = pd.read_csv('all.csv')
columns = list(df.columns[:-1])
X = df.iloc[:,:75].values
y = df.iloc[:,75].values

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

knn = KNeighborsClassifier(n_neighbors=4)
sfs1 = SFS(knn, 
           k_features=4, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=0)

#sfs1 = sfs1.fit(X, y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_norm = mms.fit_transform(X)
chi = chi2(X_norm, y)[1]
f_test = f_classif(X, y)[1]
mutual = mutual_info_classif(X, y)

dt = DecisionTreeClassifier().fit(X,y)
rf = RandomForestClassifier(n_estimators=100).fit(X,y)

tree = dt.feature_importances_
forest = rf.feature_importances_

scores = [chi, f_test, mutual, tree, forest]
norm_scores = []
for s in scores:
    mms = MinMaxScaler()
    s = s.reshape(-1,1)
    s = mms.fit_transform(s)
    norm_scores.append(s)
    
X_score = np.concatenate((norm_scores), axis=1)

sensor_1 = []
sensor_2 = []
sensor_3 = []
sensor_4 = []
sensor_5 = []

sum_scores = []
for i in range(X_score.shape[0]):
    sum_scores.append(X_score[i,:].mean())
sum_scores = np.array(sum_scores)

for i in range(15):
    sensor_1.append(sum_scores[i])
    sensor_2.append(sum_scores[i+15])
    sensor_3.append(sum_scores[i+30])
    sensor_4.append(sum_scores[i+45])
    sensor_5.append(sum_scores[i+60])

sensor_1 = tuple(sensor_1)
sensor_2 = tuple(sensor_2)
sensor_3 = tuple(sensor_3)
sensor_4 = tuple(sensor_4)
sensor_5 = tuple(sensor_5)

ind = np.arange(15)
width=0.15
alpha = 0.8
fig, ax = plt.subplots()
rects1 = ax.bar(ind, sensor_1, width, color='#DC143C', alpha=alpha)
rects2 = ax.bar(ind + width, sensor_2, width, color='#228B22', alpha=alpha)
rects3 = ax.bar(ind + 2*width, sensor_3, width, color=(0.1, 0.2, 0.5), alpha=alpha)
rects4 = ax.bar(ind + 3*width, sensor_4, width, color='#FF8C00', alpha=alpha)
rects5 = ax.bar(ind + 4*width, sensor_5, width, color='c', alpha=alpha)

ax.set_title('Feature Importances for Sensor Positions')
ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),\
          ('Left Leg', 'Left Arm', 'Chest', 'Right Arm', 'Right Leg'),\
          prop={'size': 14})
ax.set_ylabel('Mean Probability', size=16)
ax.set_xticks(ind + width*2)
ax.set_xticklabels(('Frequency', 'Magnitude', 'AC Lag 1', 'AC Lag 2', 'AC Lag 3', 'AC Lag 4',\
                    'AC Lag 5', 'AC Lag 6', 'AC Lag 7', 'AC Lag 8', 'AC Lag 9', 'Engery',\
                    'Abs Sum Diff', 'Skewness', 'Kurtosis'), rotation='vertical')
plt.show()

#splits into 2 plots

ind = np.arange(8)
width = 0.15
alpha = 0.8
fig, ax = plt.subplots()
rects1 = ax.bar(ind, sensor_1[:8], width, color='r', alpha=0.8)
rects2 = ax.bar(ind + width, sensor_2[:8], width, color='y', alpha=alpha)
rects3 = ax.bar(ind + 2*width, sensor_3[:8], width, color='b', alpha=alpha)
rects4 = ax.bar(ind + 3*width, sensor_4[:8], width, color='g', alpha=alpha)
rects5 = ax.bar(ind + 4*width, sensor_5[:8], width, color='c', alpha=alpha)

ax.set_xticks(ind + width*2)
ax.set_xticklabels(('Frequency', 'Magnitude', 'AC Lag 1', 'AC Lag 2', 'AC Lag 3', 'AC Lag 4',\
                    'AC Lag 5', 'AC Lag 6'), rotation='vertical')
plt.show()

ind = np.arange(9)
width = 0.15
alpha = 0.8
fig, ax = plt.subplots()
rects1 = ax.bar(ind, sensor_1[8:], width, color='r', alpha=0.8)
rects2 = ax.bar(ind + width, sensor_2[8:], width, color='y', alpha=alpha)
rects3 = ax.bar(ind + 2*width, sensor_3[8:], width, color='b', alpha=alpha)
rects4 = ax.bar(ind + 3*width, sensor_4[8:], width, color='g', alpha=alpha)
rects5 = ax.bar(ind + 4*width, sensor_5[8:], width, color='c', alpha=alpha)

ax.set_xticks(ind + width*2)
ax.set_xticklabels(('AC Lag 7', 'AC Lag 8', 'AC Lag 9', 'Welch 2', 'Welch 5', 'Welch 8',\
                    'Abs Engery', 'Skewness', 'Kurtosis', ), rotation='vertical')
plt.show()
