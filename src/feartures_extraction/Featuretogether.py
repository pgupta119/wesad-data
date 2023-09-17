import pandas as pd
import numpy as np
import scipy
import scipy.signal as scisig


class DataProcessor:

    def __init__(self, data_paths):
        """ Initialize the DataProcessor with various data paths.

        Parameters:
        - data_paths (dict): Dictionary containing paths for chest, wrist, and readme data.
        """
        self.data_paths = data_paths
        self.data = {}
        self.merged_data = None

    def load_data(self):
        """Load the datasets."""
        for key, path in self.data_paths.items():
            self.data[key] = pd.read_csv(path)  # Assuming the data is in CSV format.

    def merge_data(self):
        """Merge the datasets based on a common key, ensuring that personal information remains consistent."""
        # Assuming 'subject_id' is the common column across all datasets.
        # This also assumes that the data from different sensors for a single subject has the same 'subject_id'
        self.merged_data = self.data['chest'].merge(self.data['wrist'], on='subject_id', how='outer')
        self.merged_data = self.merged_data.merge(self.data['readme'], on='subject_id', how='outer')

    def process(self):
        """Execute the full processing."""
        self.load_data()
        self.merge_data()
        return self.merged_data


# Usage:
data_paths = {
    'chest': '/path_to_chest_data.csv',
    'wrist': '/path_to_wrist_data.csv',
    'readme': '/path_to_readme_data.csv'
}

processor = DataProcessor(data_paths)
merged_data = processor.process()
print(merged_data.head())
