import pandas as pd


class WESADResampler:
    def __init__(self, fs_dict, column_names):
        self.fs_dict = fs_dict
        self.column_names = column_names
        # self.max_frequency = max(self.fs_dict.values())
        self.dfs = {}

    def _create_dataframe(self, data, key):
        df_temp = pd.DataFrame(data[key], columns=self.column_names[key])
        timedelta_index = pd.to_timedelta((1 / self.fs_dict[key]) * df_temp.index, unit='s')
        df_temp.index = timedelta_index
        return df_temp

    def resample_data(self, data, wrist_data_):
        for key in self.column_names.keys():
            data_source = data if key == 'label' else wrist_data_
            df_temp = self._create_dataframe(data_source, key)
            self.dfs[key] = df_temp

        df = self._join_dataframes()
        # df['EDA'] = df['EDA'].fillna(method='bfill')
        df['label'] = df['label'].fillna(method='bfill')
        df = self.filter_labels(df)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def filter_labels(data):
        labels_to_remove = [0, 4, 5, 6, 7]
        for label in labels_to_remove:
            data = data[data['label'] != label]
        return data

    def _join_dataframes(self):
        # Start with the first key in the list and join the rest
        base_df = self.dfs[list(self.column_names.keys())[0]]
        for key in list(self.column_names.keys())[1:]:
            base_df = base_df.join(self.dfs[key], how='outer')

        return base_df

