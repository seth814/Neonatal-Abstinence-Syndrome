'''
The master decomposition file used to decompose raw accelerometer data.
The following steps are executed in this order:
    Magnitudes from three axes of accelerometers are calculated
    Magnitudes are highpass filtered at 1 hertz
    RMS values are calculated and a motion threshold slightly\
    above the noise floor is created
    Data is transformed in an fft 200 points (10 seconds) at a time
    All the remaining tsfresh decomposition source code is ran
    https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
    Decomposed datasets are saved for each baby
'''

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

class noise:
    def __init__(self, n):
        
        self.df_comp = pd.read_csv(str(n) + '.csv')
        self.cols = ['LL','LA','C','RA','RL']
        self.df = pd.DataFrame()
        #build empty fft dataframe
        for t in ['M_', 'F_']:
            for c in self.df.columns:
                self.df_fft[t + c] = []
        self.percent_removed = 0.0
        
        #preprocess the magnitudes and high pass filter
        self.calc_mag()
        #self.plot_comp()
        self.high_pass()
        #self.plot_high()
        #self.plot_noise_floor()
        
        #set the magnitude of the noise floor to filter noise
        self.magnitude_threshold()
        self.cycle_window()
        self.cycle_window_logic()
        self.cycle_mask()
        self.rebuild_with_mask()

        #start processing fft and tsfresh
        self.round_down()
        self.decompose()
        
    def grouped(self, iterable, n):
        "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
        return zip(*[iter(iterable)]*n)

    def calc_mag(self):
        #create magnitudes of accelerometers
        for c, (x, y, z) in zip(self.cols, self.grouped(self.df_comp.columns, 3)):
            self.df[c] = np.sqrt(np.square(self.df_comp[x]) + np.square(self.df_comp[y]) + np.square(self.df_comp[z]))
    
    def plot_comp(self):
        f, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, sharex=True, sharey=False)
        self.ax1.set_title('Accelerometer Preprocessing', size=14)
        self.ax1.plot(self.df_comp.iloc[50:150,0], label='X Component')
        self.ax1.plot(self.df_comp.iloc[50:150,1], label='Y Component')
        self.ax1.plot(self.df_comp.iloc[50:150,2], label='Z Component', color='#9400D3', alpha=0.5)
        self.ax1.legend(loc=6, prop={'size': 12}, bbox_to_anchor=(1, 0.5))
        plt.gca().axes.get_xaxis().set_ticks([])
        
        self.ax2.set_ylabel('Magnitude (g)', size=14)
        self.ax2.plot(self.df.iloc[50:150,0], label= 'Accelerometer\nMagnitude', color='#FF8C00', alpha=0.5)
        self.ax2.legend(loc=6, prop={'size': 12}, bbox_to_anchor=(1, 0.5))
        
    def plot_high(self):
        self.ax3.plot(self.df.iloc[50:150,0], label='Highpassed\nMagnitude', color='c')
        self.ax3.legend(loc=6, prop={'size': 12}, bbox_to_anchor=(1, 0.5))
        plt.show()
        
    def plot_noise_floor(self):
        plt.title('Accelerometer Noise Floor', size=14)
        plt.plot(self.df.iloc[220:320,0].values, label='Highpassed')
        signal = self.df.iloc[220:320,0].values
        rms = []
        for i in range(0, 100, 1):
            rms.append(np.sqrt(np.mean(np.square(signal[i:i+20]))))
        plt.plot(rms, label='RMS')
        plt.legend(loc=6, prop={'size': 12}, bbox_to_anchor=(1, 0.5))
        plt.ylabel('Magnitude (g)', size=14)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.show()
    
    def calc_rms(self, signal):
        #calculates rms with 20 point window
        rms = []
        for i in range(signal.shape[0]):
            rms.append(np.sqrt(np.mean(np.square(signal[i:i+20]))))
        return np.array(rms)
    
    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    
    def high_pass(self):
        for c in self.df.columns:
            filtered = self.butter_highpass_filter(self.df[c], 1, 20)    #data, cutoff, sampling freq
            self.df[c] = filtered
    
    def visualize(self):
        for c in self.df.columns:
            plt.plot(self.df[c])
            plt.show()
    
    def magnitude_threshold(self):
        self.bool = []
        for c in self.df.columns:
            a  = np.array(self.df[c])
            a = self.calc_rms(a)
            self.bool.append(a < 0.008)
        
    def cycle_window(self):
        #looks at a window of 100 points
        self.window = []
        for array in self.bool:
            a = self.rolling_window(array, 100)
            self.window.append(a)
        
    def cycle_window_logic(self):
        self.logic = []
        for array in self.window:
            a = self.logical_all(array)
            self.logic.append(a)
    
    def cycle_mask(self):
        logic = np.array(self.logic).T
        mask = []
        for row in logic:
            a = np.all(row)
            a = np.logical_not(a)
            mask.append(a)
        self.mask = np.array(mask)
    
    def rebuild_with_mask(self):
        df = pd.DataFrame(columns=self.df.columns)
        old_len = self.df.shape[0]
        for c in self.df.columns:
            df[c] = self.df[c].values[self.mask]
        self.df = df
        new_len = self.df.shape[0]
        percentage = (old_len - new_len)/float(old_len)
        percentage = round(percentage, 4) * 100
        self.percent_removed = "{0:.2f}".format(percentage)
        print('Removed {0:.2f}% of data believed to be noise!'.format(round(percentage, 2)))
    
    def rolling_window(self, y, n):
        window = []
        for i in range(0, y.shape[0], 1):
            s = y[i:i+n]
            window.append(s)
        return window
    
    def logical_all(self, y):
        result = []
        for i in y:
            r = np.all(i)
            result.append(r)
        result = np.array(result)
        return result

    def round_down(self):
        num = self.df.shape[0]
        divisor = 200
        length = num - (num%divisor)
        self.df = self.df.iloc[:length, :]
    
    def fft(self, y, c):
        n = len(y)                       # length of the signal
        freq = np.fft.fftfreq(n, d=1/20)
        #freq = frq[range(n/2)]          # one side frequency range
                                         # removed this to replace with a more accurate step size
        Y = np.fft.fft(y)/n              # fft computing and normalization
        Y = Y[:n//2]
        Y = abs(Y)
        freq = freq[:n//2]
        
        #uncomment to plot
        '''
        t = np.arange(len(y)) * 0.05
        f, axarr = plt.subplots(2)
        axarr[0].plot(t, y)
        axarr[0].set_title('Time and Frequency Domains', size=14)
        axarr[0].set_ylabel('Magnitude (g)', size=14)
        axarr[1].plot(freq, Y)
        axarr[1].set_title('Time (s)', size=14)
        axarr[1].set_xlabel('Frequency (Hz)', size=14)
        axarr[1].set_ylabel('Norm Magnitude', size=14)
        plt.tight_layout()
        plt.show()
        '''
        
        max_mag = max(Y)
        index = np.where(Y == max_mag)[0]
        freq = float(freq[index])
        df_fft = pd.DataFrame({c+'_mag':[max_mag], c+'_freq':[freq]})

        return df_fft
        
    def autocorrelation(self, x, c):
        """
        Calculates the lag autocorrelationelation of a lag value of lag.
    
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param lag: the lag
        :type lag: int
        :return: the value of this feature
        :return type: float
        """
        x = pd.Series(x)
        lag = []
        for l in range(1,10,1):
            lag.append(pd.Series.autocorr(x, l))
        df_auto = pd.DataFrame({c+ '_autocorrelation_lag_1':[lag[0]],\
                                c+ '_autocorrelation_lag_2':[lag[1]], c+ '_autocorrelation_lag_3':[lag[2]],\
                                c+ '_autocorrelation_lag_4':[lag[3]], c+ '_autocorrelation_lag_5':[lag[4]],\
                                c+ '_autocorrelation_lag_6':[lag[5]], c+ '_autocorrelation_lag_7':[lag[6]],\
                                c+ '_autocorrelation_lag_8':[lag[7]], c+ '_autocorrelation_lag_9':[lag[8]]})
        return df_auto

    def absolute_sum_of_changes(self, x, c):
        """
        Returns the sum over the absolute value of consecutive changes in the series x
    
        .. math::
    
            \\sum_{i=1, \ldots, n-1} \\mid x_{i+1}- x_i \\mid
    
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        
        abs_sum = np.sum(abs(np.diff(x)))
        df_abs_sum = pd.DataFrame({c+'_abs_sum_of_changes':[abs_sum]})
        
        return df_abs_sum

    def abs_energy(self, x, c):
        """
        Returns the absolute energy of the time series which is the sum over the squared values
    
        .. math::
    
            E = \\sum_{i=1,\ldots, n} x_i^2
    
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        x = np.asarray(x)
        df_energy = pd.DataFrame({c+'_abs_energy':[sum(x*x)]})
        
        return df_energy

    def skewness(self, x, c):
        
        x = pd.Series(x)
        df_shew = pd.DataFrame({c+'_skew':[pd.Series.skew(x)]})
        return df_shew
    
    def kurtosis(self, x, c):
        """
        Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
        moment coefficient G2).
    
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        x = pd.Series(x)
        df_kurtosis = pd.DataFrame({c+'_kurtosis':[pd.Series.kurtosis(x)]})
        
        return df_kurtosis

    def decompose(self):
        #crawl through data 200 points (10 seconds) at a time
        frames_vert = []
        for i in range(0, self.df.shape[0], 200):
            frames_hori = []
            for c in self.cols:
                s = self.df[c][i:i+200]  #signal
                frames_hori.append(self.fft(s, c))
                frames_hori.append(self.autocorrelation(s, c))
                frames_hori.append(self.abs_energy(s, c))
                frames_hori.append(self.absolute_sum_of_changes(s, c))
                frames_hori.append(self.skewness(s, c))
                frames_hori.append(self.kurtosis(s, c))
                
            row = pd.concat((frames_hori), axis=1)
            frames_vert.append(row)
        self.df_decomp = pd.concat((frames_vert), axis=0)


    def get_dataframe(self):
        return self.df
    
    def get_decomp(self):
        return self.df_decomp

    def get_percent(self):
        return self.percent_removed

#set plot settings
plt.rc('figure', figsize=(10,6))
#plt.rc('font', size=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=10)

healthy = [44,49,54,55,57,70,72,77]   #healthy
nas = [74,88,76,62,86,78,58,66]       #nas

path = os.getcwd()
h_path = path + '/non_datasets'
n_path = path + '/nas_datasets'
h_decomp = path + '/non_decomp'
n_decomp = path + '/nas_decomp'

h = (healthy, h_path, h_decomp)
g = (nas, n_path, n_decomp)


for n in h[0]:
    os.chdir(h[1])
    print('Processing infant number {}'.format(n))
    f = noise(n)
    df = f.get_dataframe()
    df_decomp = f.get_decomp()
    os.chdir(h[2])
    df_decomp.to_csv(str(n)+'_decomp.csv', index=None)

for n in g[0]:
    os.chdir(g[1])
    print('Processing infant number {}'.format(n))
    f = noise(n)
    df = f.get_dataframe()
    df_decomp = f.get_decomp()
    os.chdir(g[2])
    df_decomp.to_csv(str(n)+'_decomp.csv', index=None)