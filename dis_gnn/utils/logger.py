from datetime import datetime
import os
import shutil
def logger(path, value, stamp, desc=None, config_path = './dis_gnn/config/config.yaml'):
    # Ensure desc has a valid filename
    if desc is None:
        desc = "log.txt"
    elif not desc.endswith(".txt"):
        desc += ".txt"
    
    # Create a timestamped root folder
    root_path = os.path.join(path, stamp)
    os.makedirs(root_path, exist_ok=True)
    
    # Path to the log file
    file_path = os.path.join(root_path, desc)
    
    # Write the value to the file
    with open(file_path, "a") as f:
        f.write(str(value) + "\n")
    
    shutil.copy(config_path, root_path)
