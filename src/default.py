import pandas as pd
import os

path = "../classification/"

def show_as_dataframe(frame):
    return pd.DataFrame(frame)

def default_classification_dataset(datasetname):
    files = os.listdir(path)
    names = ["cancer", "diabetics", "wine", "digit"]
    if datasetname == "cancer":
        df = pd.read_csv(os.path.join(path, files[0]))
    elif datasetname == "diabetics":
        df = pd.read_csv(os.path.join(path, files[1]))
    elif datasetname == "digit":
        df = pd.read_csv(os.path.join(path, files[2]))
    elif datasetname == "iris":
        df = pd.read_csv(os.path.join(path, files[3]))
    elif datasetname == "wine":
        df = pd.read_csv(os.path.join(path, files[4]))
    else:
        print(f"Invalid dataset name.\nTry Those {names}")
        
print(default_classification_dataset("cancer"))