import pandas as pd
import os

# Path to the base directory
base_path = "C:\\Users\\Aweso\\Downloads\\The folder\\Data\\DB1"

# Creating the columns we want to transfer
columns = [f'emg{x}' for x in range(10)]
columns.extend(["restimulus", "repetition"])
DBCollector = {}
BigDB = pd.DataFrame()

# Looping through all the files in DB1 and copying the subset to a new directory
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".csv") and 'E3' in file:
            full_file_path = os.path.join(root, file)
            new_file_path = os.path.join(base_path, 'E3Data.csv')

            df = pd.read_csv(full_file_path)  # reading from the preprocessed dataset

            # Creating a new dataframe with the selected columns from df
            new_df = df[columns]

            dataframe_name = os.path.splitext(file)[0]  # Use the file name (without extension) as the key
            DBCollector[dataframe_name] = new_df

# Combining all dataframes in DBCollector into a single dataframe
combined_df = pd.concat(DBCollector.values(), ignore_index=True)

# Saving the combined dataframe to a new CSV file
combined_df.to_csv(new_file_path, index=False)
print(f"Combined CSV file saved to {new_file_path}")
print(combined_df.head(1000))
