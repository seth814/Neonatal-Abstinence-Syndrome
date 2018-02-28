'''
Imports the saved results from lstm_optimization.
Plots learning and validation curves.
'''

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats

def plot_univariate(metric_dict,
                    label,
                    kind='std_dev',
                    color='blue',
                    bcolor='steelblue',
                    marker='o',
                    alpha=0.2):
        
    #set plot settings
    plt.rc('figure', figsize=(10,6))
    plt.rc('font', size=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    
    """Plot feature selection results.
    Parameters
    ----------
    metric_dict : mlxtend.SequentialFeatureSelector.get_metric_dict() object
    kind : str (default: "std_dev")
        The kind of error bar or confidence interval in
        {'std_dev', 'std_err', 'ci', None}.
    color : str (default: "blue")
        Color of the lineplot (accepts any matplotlib color name)
    bcolor : str (default: "steelblue").
        Color of the error bars / confidence intervals
        (accepts any matplotlib color name).
    marker : str (default: "o")
        Marker of the line plot
        (accepts any matplotlib marker name).
    alpha : float in [0, 1] (default: 0.2)
        Transparency of the error bars / confidence intervals.
    """
    
    k_feat = sorted(metric_dict.keys())
    avg = [metric_dict[k]['avg_score'] for k in k_feat]
    upper, lower = [], []

    for k in k_feat:
        upper.append(metric_dict[k]['avg_score'] +
                     metric_dict[k][kind])
        lower.append(metric_dict[k]['avg_score'] -
                     metric_dict[k][kind])
    
    plt.plot(k_feat, avg, label=label, color=color, marker=marker)
    plt.fill_between(k_feat, upper, lower, alpha=alpha, color=bcolor, lw=1)
    plt.title('Learning Curve for 5 Epoch LSTM')
    plt.ylabel('Accuracy w. Std.Dev.')
    plt.xlabel('Fraction of Data Size')
    plt.xticks(k_feat)
    plt.ylim([0.68,0.8])
    plt.grid()
    plt.legend()

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
os.chdir(path+'/pickles')

with open('node.pickle', 'rb') as handle:
    nodes = pickle.load(handle)
with open('learning_curve.pickle', 'rb') as handle:
    learn = pickle.load(handle)

metric_dict = nodes
#plot_univariate()
train = {}
test = {}

for key, value in learn.items():
    stats = calc_stats(learn[key]['train_cv'])
    train[key] = stats
    stats = calc_stats(learn[key]['test_cv'])
    test[key] = stats
    
plot_univariate(train, label='Train', color='blue', bcolor='steelblue')
plot_univariate(test, label='Test', color='red', bcolor='indianred')
plt.style.use('ggplot')
plt.show()