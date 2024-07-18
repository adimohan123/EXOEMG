import h5py
import pandas as pd

# Open the HDF5 file
file_path = "C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1_S1_image.h5"
with h5py.File(file_path, 'r') as h5file:
    # List all keys (dataset names) in the file
    keys = list(h5file.keys())
    print("Keys: ", keys)

data = pd.read_hdf("C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1_S1_image.h5",'imageLabel')
print(data)