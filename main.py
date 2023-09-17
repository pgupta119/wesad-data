from src.prerpocessing.wristpreprocessing import DataProcessor
from src.prerpocessing.upsampling import WESADResampler
from src.feartures_extraction.Featureextractionwrist import WESADProcessor
from src.models.ML_model import WESADLDA
import pandas as pd
import pickle

if __name__ == '__main__':
    # preprocessing for wrist
    processor = DataProcessor("/Users/prakashgupta/oneprojectassignment/data/S2/S2.pkl")
    processor.load_data()
    wrist_data = processor.process_data()

    # upsampling for wrist
    fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
    # with open('/Users/prakashgupta/oneprojectassignment/parse/data.pkl', 'rb') as f:
    #     wrist_data = pickle.load(f, encoding='latin1')
    with open('/Users/prakashgupta/oneprojectassignment/data/S2/S2.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    column_names = {
        'ACC': ['ACC_x', 'ACC_y', 'ACC_z'],
        'BVP': ['BVP'],
        'EDA': ['EDA'],
        'TEMP': ['TEMP'],
        'label': ['label']
    }
    # Assuming 'data' and 'wrist_data_' are already defined
    resampler = WESADResampler(fs_dict, column_names)
    result = resampler.resample_data(data, wrist_data)

    # feature extraction for wrist
    # Get the samples from your feature extraction methods
    fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}

    data = result
    data['label'] = data['label'].astype(int)
    processor = WESADProcessor(data, fs_dict, 60)  # Window size changed to 60 seconds
    samples = processor.process()
    wrist_samples = samples
    all_samples = pd.concat([wrist_samples])
    # print(all_samples)
    # model training for wrist
    model = WESADLDA()
    X_train, X_test, y_train, y_test = model.transform(all_samples)
