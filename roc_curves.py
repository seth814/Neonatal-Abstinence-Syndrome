'''
Plots Reciever Operator Characteristics for all ml models
'''

import os
import pickle
from sklearn.metrics import auc
import matplotlib.pyplot as plt

#set plot settings
plt.rc('figure', figsize=(10,8))
plt.rc('font', size=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.style.use('ggplot')

path = os.getcwd()
os.chdir(path+'/pickles')
with open('roc_ml.pickle', 'rb') as handle:
    roc_ml = pickle.load(handle)
with open('roc_mlp.pickle', 'rb') as handle:
    roc_mlp = pickle.load(handle)
with open('roc_rnn.pickle', 'rb') as handle:
    roc_rnn = pickle.load(handle)
  
colors = ['#DC143C', '#FF8C00', (0.1, 0.2, 0.5), '#228B22', 'c']
for (key, value), c in zip(sorted(roc_ml.items()), colors):
    fpr = value['fpr']
    tpr = value['tpr']
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, label='%s (auc = %0.2f)' % (key, roc_auc), color=c)

plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2,
         label='random guessing')
plt.plot([0, 0, 1],
         [0, 1, 1],
         lw=2,
         linestyle=':',
         color='black',
         label='perfect performance')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.title('Receiver Operator Characteristics', size=16)
plt.xlabel('False Positive Rate', size=16)
plt.ylabel('True Positive Rate', size=16)
plt.legend(loc='lower right', prop={'size': 14})
plt.show()

#neural net rocs
roc_neural = dict(roc_mlp, **roc_rnn)

for key, value in sorted(roc_neural.items()):
    fpr = value['fpr']
    tpr = value['tpr']
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, label='%s (auc = %0.2f)' % (key, roc_auc))
    
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2,
         label='random guessing')
plt.plot([0, 0, 1],
         [0, 1, 1],
         lw=2,
         linestyle=':',
         color='black',
         label='perfect performance')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.title('Receiver Operator Characteristics', size=16)
plt.xlabel('False Positive Rate', size=16)
plt.ylabel('True Positive Rate', size=16)
plt.legend(loc='lower right', prop={'size': 14})
plt.show()