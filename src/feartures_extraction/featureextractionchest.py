import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle
import os
import scipy.signal as scisig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class WESADProcessor:
    def __init__(self, data, fs_dict, window_in_seconds):
        self.data = data
        self.fs_dict = fs_dict
        self.window_in_seconds = window_in_seconds

    @staticmethod
    def filter_labels(data):
        labels_to_remove = [0, 4, 5, 6, 7]
        for label in labels_to_remove:
            data = data[data['label'] != label]
        return data

    def process(self):
        self.data = self.filter_labels(self.data)
        grouped = self.data.groupby('label')
        baseline = grouped.get_group(1)
        stress = grouped.get_group(2)
        amusement = grouped.get_group(3)

        baseline_samples = self.get_samples(baseline, 1)
        stress_samples = self.get_samples(stress, 2)
        amusement_samples = self.get_samples(amusement, 3)

        all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
        return all_samples

    def get_samples(self, data, label):
        n_windows = int(len(data) / (self.fs_dict['label'] * self.window_in_seconds))
        samples = []
        window_len = self.fs_dict['label'] * self.window_in_seconds
        for i in range(n_windows):
            w = data[window_len * i: window_len * (i + 1)]
            w = pd.concat([w, self.get_net_accel(w)])
            cols = list(w.columns)
            cols[0] = 'net_acc'
            w.columns = cols

            wstats = self.get_window_stats(data=w, label=label)
            x = pd.DataFrame(wstats).drop('label', axis=0)
            y = x['label'][0]
            x.drop('label', axis=1, inplace=True)
            wdf = pd.DataFrame(x.values.flatten()).T
            wdf = pd.concat([wdf, pd.DataFrame({'label': y}, index=[0])], axis=1)
            wdf['BVP_peak_freq'] = self.get_peak_freq(w['BVP'].dropna())
            wdf['TEMP_slope'] = self.get_slope(w['TEMP'].dropna())
            samples.append(wdf)
        return pd.concat(samples)

    @staticmethod
    def get_slope(series):
        linreg = scipy.stats.linregress(np.arange(len(series)), series)
        return linreg[0]

    @staticmethod
    def get_window_stats(data, label=-1):
        mean_features = np.mean(data)
        std_features = np.std(data)
        min_features = np.amin(data)
        max_features = np.amax(data)
        features = {'mean': mean_features, 'std': std_features, 'min': min_features, 'max': max_features,
                    'label': label}
        return features
    # ... (original function code)

    @staticmethod
    def get_net_accel(data):
        return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))
    # ... (original function code)

    @staticmethod
    def get_peak_freq(x):
        f, Pxx = scisig.periodogram(x, fs=8)
        psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
        peak_freq = psd_dict[max(psd_dict.keys())]
        return peak_freq

# ... (original function code)




if __name__ == "__main__":
    fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
    with open('/Users/prakashgupta/oneprojectassignment/parse/upsampling.pkl', 'rb') as file:
        data = pd.read_pickle(file)
    data['label'] = data['label'].astype(int)
    processor = WESADProcessor(data, fs_dict, 30)
    samples = processor.process()
