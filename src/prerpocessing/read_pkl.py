import os
import pickle
import re


def read_pkl_file(filepath):
    """Read a .pkl file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            return data
    except Exception as e:
        print(f"Failed to read {filepath}. Reason: {e}")
        return None


class PickleReader:
    def __init__(self, base_directory):
        """
        :type base_directory: object
        """
        self.base_directory = base_directory

    def find_pkl_files(self, path=None):
        """Find all .pkl files recursively that match the pattern."""
        if path is None:
            path = self.base_directory

        pkl_files = []
        pattern = re.compile(r"^[a-zA-Z][0-9].*\.pkl$")

        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                if pattern.match(filename):
                    pkl_files.append(os.path.join(dirpath, filename))

        return pkl_files

    def process_all_files(self):
        """Read all matching .pkl files."""
        # all_pkl_files = self.find_pkl_files()
        all_pkl_files = ['/Users/prakashgupta/oneprojectassignment/data/S2/S2.pkl']
        data_list = []

        for file in all_pkl_files:
            data = read_pkl_file(file)
            if data is not None:
                data_list.append(data)

        return data_list


if __name__ == "__main__":
    directory = "/Users/prakashgupta/oneprojectassignment/data"  # starting directory, change accordingly
    reader = PickleReader(directory)
    subject = ['data/S2/S2.pkl']
    all_data = reader.process_all_files()
    # print(all_data)

    # for data in all_data:
    #     print(data)
