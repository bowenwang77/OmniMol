import os
import numpy as np
import pandas as pd
import pickle
import json

# Define the subdirectory path
subdir_path = "example"

# Get a list of all the CSV file names in the subdirectory, sorted alphabetically
csv_files = sorted([f for f in os.listdir(subdir_path) if f.endswith(".csv")])

# Create the task dictionary
task_dict = {}

# Function to determine if a CSV file is a regression task
def is_regression_task(df):
    
    # Get the column names that are not "smiles" or "group"
    col = [c for c in df.columns if c not in ["smiles", "group"]][0]
    
    # Loop through each column and check if any values are not integers

    for val in df[col]:
        if pd.isna(val):
            continue
        elif not isinstance(val, (int, np.integer)):
            if not (isinstance(val, float) and val.is_integer()):
                return True,1
    #class_num = len(set(df[col].astype(int))) # A phenomenon: CL have 24 classes, but the max class num is 24
    class_num = df[col].max().astype(int)+1
    return False, class_num

# Loop through each CSV file
for csv_file in csv_files:
    # Determine if the file is a regression task
    max_class_num = 0
    # Read the CSV file using pandas
    df = pd.read_csv(os.path.join(subdir_path, csv_file))
    is_reg,class_num = is_regression_task(df)
    task_name = csv_file.split(".csv")[0]
    train = df[df.group=='training']
    if is_reg:
        task_mean = train[task_name].mean()
        task_std = train[task_name].std()
    elif task_name != 'CL':
        task_mean = train[task_name].mean()
        task_std = 1
    else:
        task_mean = 0
        task_std = 1

    if class_num > max_class_num:
        max_class_num = class_num
    print(task_name, "class_num",class_num, "mean:", task_mean, "std:", task_std)
    
    # Create the embedding 1-hot numpy array
    embedding_array = np.zeros(len(csv_files) + 1)
    embedding_array[csv_files.index(csv_file)] = 1
    task_idx = csv_files.index(csv_file)

    # Add the embedding and regression to the task dictionary
    task_dict[csv_file.split(".csv")[0]] = {
        "regression": is_reg,
        "task_idx": int(task_idx),  # Convert numpy.int64 to int
        "cls_num": int(class_num),  # Convert numpy.int64 to int
        "task_mean": float(task_mean),  # Convert numpy.float64 to float
        "task_std": float(task_std)  # Convert numpy.float64 to float
    }

# Save the task dictionary to a pickle file
# with open("./task_dictionary_mean_std_BBBP_BACE_OCP.pkl", "wb") as f:
#     pickle.dump(task_dict, f)
# import pdb
# pdb.set_trace()
with open("./meta_dict.json", 'w') as outfile:
    json.dump(task_dict, outfile, indent=4)  # 'indent=4' for pretty printing