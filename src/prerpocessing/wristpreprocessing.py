import pickle

import pandas as pd
import scipy.signal as scisig
from scipy.signal import butter, lfilter, savgol_filter


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

    def process_data(self):
        wrist_data = self.data['signal']['wrist']

        # Preprocess ACC

        acc_df = pd.DataFrame(wrist_data['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
        acc_df['ACC_x'] = self.apply_FIR_filter(acc_df['ACC_x'])
        acc_df['ACC_y'] = self.apply_FIR_filter(acc_df['ACC_y'])
        acc_df['ACC_z'] = self.apply_FIR_filter(acc_df['ACC_z'])
        wrist_data['ACC'] = acc_df[['ACC_x', 'ACC_y', 'ACC_z']].values

        # Preprocess BVP
        wrist_data['BVP'] = butter_bandpass_filter(wrist_data['BVP'], 0.7, 3.7, self.fs_dict['ACC'])

        # Preprocess EDA
        wrist_data['EDA'] = butter_lowpass_filter(wrist_data['EDA'], 1, self.fs_dict['ACC'], order=6)

        # Preprocess TEMP
        wrist_data['TEMP'] = smooth_temp_signal(wrist_data['TEMP'])
        with open('/Users/prakashgupta/WESAD_Project/data/processed/wrist_data.pkl', 'wb') as file:
            pickle.dump(wrist_data, file)

        return wrist_data

    def apply_FIR_filter(self, data, cutoff=0.4, numtaps=64):
        f = cutoff / (self.fs_dict['ACC'] / 2.0)
        FIR_coeff = scisig.firwin(numtaps, f)
        return scisig.lfilter(FIR_coeff, 1, data)


# Preprocessing utility functions:

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def smooth_temp_signal(temp_data):
    return savgol_filter(temp_data, window_length=11, polyorder=3, mode='nearest')


