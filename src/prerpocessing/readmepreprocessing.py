import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:
    def __init__(self, input_file):
        """
        Initialize DataPreprocessor with file paths.

        Parameters:
        - input_file (str): Path to input CSV file.
        - output_file (str): Path to save preprocessed data.
        """
        self.input_file = input_file
        self.data = None
        self.onehotencoder = OneHotEncoder(
            drop='first', sparse=False)  # Using drop='first' to get k-1 dummies out of k categorical levels.

    def load_data(self):
        """Load data from CSV file."""
        self.data = pd.read_csv(self.input_file)

    def get_encoded_feature_names(self, categorical_cols):
        """Construct feature names for encoded columns."""
        try:
            return self.onehotencoder.get_feature_names_out(categorical_cols)
        except AttributeError:
            return self.onehotencoder.get_feature_names(categorical_cols)

    def preprocess(self):
        """Preprocess the data: Convert categorical columns to one-hot encoding."""
        categorical_cols = ['gender', 'dominant_hand', 'coffee_today', 'coffee_last_hour', 'sport_today', 'smoker',
                            'smoke_last_hour', 'feel_ill_today']

        # One-hot encoding for categorical columns
        encoded_data = self.onehotencoder.fit_transform(self.data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=self.get_encoded_feature_names(categorical_cols))

        # Concatenate original data and encoded data
        self.data = pd.concat([self.data.drop(categorical_cols, axis=1), encoded_df], axis=1)
        print(self.data.head())
        return self.data

    def execute(self):
        """Execute the complete preprocessing pipeline."""
        self.load_data()
        self.preprocess()

