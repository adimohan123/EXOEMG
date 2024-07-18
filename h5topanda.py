import h5py
import pandas as pd

file = h5py.File("C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1_S1_image.h5", 'r')
imageData = file['imageData'][:]
imageLabel = file['imageLabel'][:]
file.close()
print(imageData.shape)
print(imageLabel.shape)

df = pd.DataFrame(imageLabel)
print(df)
'''
# Specify the HDF5 file path
hdf5_file_path = r'C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1_S1_image.h5'

# Use a context manager to handle the HDF5 file
with pd.HDFStore(hdf5_file_path) as hdf_store:
    # Load the keys from the HDF5 file
    keys = hdf_store.keys()
    print (keys)

    # Convert each table to CSV
    for key in keys:
        # Remove the leading '/' from the key if it exists
        key = key.lstrip('/')

        # Load the table into a pandas DataFrame
        df = pd.read_hdf(hdf5_file_path, key)

        # Save the DataFrame to a CSV file
        csv_file_path = hdf5_file_path.replace('DB1_S1_image.h5', f'{key}.csv')
        df.to_csv(csv_file_path, index=False)
        print('hello')

        print(f"Converted {hdf5_file_path} (key: {key}) to {csv_file_path}")
'''
