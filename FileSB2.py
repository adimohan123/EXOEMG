import os
import scipy.io
import pandas as pd
import numpy as np

import os
import scipy.io
import pandas as pd
import numpy as np


def convert_mat_to_csv(directory):
    """Convert all .mat files in the specified directory to .csv files."""
    for file in os.listdir(directory):
        if file.endswith('.mat'):
            mat_file_path = os.path.join(directory, file)
            newdir = os.path.join(directory, "CSV")
            if not os.path.exists(newdir):
                os.makedirs(newdir)
            csv_file_path = os.path.join(newdir, file.replace('.mat', '.csv'))
            print(file)

            # Load .mat file
            mat = scipy.io.loadmat(mat_file_path)
            data = pd.DataFrame(mat['emg'])


                # Ensure the extended_restimulus matches the length of the data
            if len(mat['restimulus']) != len(data):
                print(f"Error: Length mismatch in file {file}")
                continue  # Skip this file if there's a length mismatch


                # Directly assign mat['restimulus'] to data['stimulus']
            data['stimulus'] = mat['restimulus']

            # Assign mat['repetition'] to data['repetition']
            data['repetition'] = mat['repetition']

            # Filter out rows where 'stimulus' is 0
            data = data[data['stimulus'] != 0]

            # Save DataFrame to .csv
            data.to_csv(csv_file_path, index=False)
            print(f"Converted {mat_file_path} to {csv_file_path}")



def main():
    directory = 'C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB2'  # Change this to your directory path
    convert_mat_to_csv(directory)

if __name__ == "__main__":
    main()
