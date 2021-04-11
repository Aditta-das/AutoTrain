import pandas as pd 
import numpy as np 
from encodings.aliases import aliases
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import sys, os
from threading import Thread
import warnings
warnings.filterwarnings("ignore")
'''
Step1: data load
step2: fold or not fold
step3: data clean
step4: feature and label encoding / one hot encoding
        if feature or label is str:
'''
class AutoDeepClassification:
    def __init__(self, path, label, fold):
        super(AutoDeepClassification, self).__init__()
        self.path = path
        self.label = label
        self.fold = fold
    
    def create_folds(self, df):
        df["kfold"] = -1
        df = df.sample(frac=1).reset_index(drop=True)
        y = df[self.label].values
        kf = model_selection.StratifiedKFold(n_splits=5)
        for fold, (tr_, val_) in enumerate(kf.split(X=df, y=y)):
            df.loc[val_, "kfold"] = fold
        print(f"[INFO] Create Folds Complete")
        return df

    def with_out_kfold(self, df):
        return df

    def data_clean(self, df):
        '''
        check null columns and null rows 
        if row null is less then 20% of total row fill them else drop those row
        '''
        for i, col in enumerate(df.columns):
            missing_data = df[df.columns[i]].isna().sum()
            perc = int(missing_data / len(df) * 100)
            if df.isna().any()[col] == True and perc > int(50):
                df.drop([col], axis=1, inplace=True)
                print(f"[INFO] {col} Column dropped")

            elif perc <= int(30):
                '''
                Fill the column with mean / median/ mode
                '''

                pass
        self.data_clean_extra_column(df)
        return df

    def data_clean_extra_column(self, df):
        '''
        if you want to remove extra columns
        '''
        for col in df.columns:
            col_name = input("ENTER THE COLUMN NAME YOU WANT TO REMOVE / TYPE 'quit': ")
            if col_name in df.columns:
                df.drop([col], axis=1, inplace=True)
            elif col_name == "quit":
                break
            elif col_name not in df.columns:
                print(f"[WRONG] INPUT CORRECT COLUMN NAME")

    def scale_data(self, df):

        return data

    def split_data(self, df, data_input):
        '''
        Check total unique label
        We apply One-Hot Encoding when:
            The categorical feature is not ordinal (like the countries above)
            The number of categorical features is less so one-hot encoding can be effectively applied
        We apply Label Encoding when:
            The categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school)
            The number of categories is quite large as one-hot encoding can lead to high memory consumption
        '''
        unique_label = df[self.label].unique()
        if len(unique_label) <= 3 and data_input == "one-hot":
            '''
            Create a right and suggestion type system
            if you choose one-hot:
                will say "right"
            else: will say "suggested line"
            '''
            print(f"[INFO] Suggest to apply One-Hot Encoding\n\tBecause, \n\tThe categorical feature is not ordinal\n\tThe number of categorical features is less so one-hot encoding can be effectively applied\n")            
        else:
            print(f"[INFO] Suggest to apply Label Encoding\n\tBecause, \n\tThe categorical feature is ordinal\n\tThe number of categories is quite large as one-hot encoding can lead to high memory consumption\n")

        if len(unique_label) <= 3 and df[self.label].dtypes == "object" and data_input == "one-hot":
            dummy2 = pd.get_dummies(df[self.label])
            df = pd.concat([dummy2, df], axis=1).drop(self.label, axis=1)
        else:    
            labelencode = preprocessing.LabelEncoder()
            df[self.label] = labelencode.fit_transform(df[self.label]) 
        return df
        

    def load_data(self):
        '''
        load_data_folder: give the path of the folder
        csv : return a csv file
        '''
        for encoding in set(aliases.values()):
            try:
                dataframe = pd.read_csv(self.path, encoding=encoding)
                print(f"[SUCCESS] Dataset Loaded Successfully")
                print(dataframe)
                self.data_clean(df=dataframe)   # Data cleaning null columns
                if self.fold == True: 
                    df = self.create_folds(df=dataframe)
                else: 
                    df = self.with_out_kfold(df=dataframe)

                data_input = input("ENTER THE NAME OF ENCODER: ")
                split_df = self.split_data(df, data_input)
                return split_df
            except Exception as e:
                print(f"[ERROR] {e}")

    


df = AutoDeepClassification(path="../classification/breast_cancer.csv", label="diagnosis", fold=True)
a = df.load_data()
print(a)

# df = pd.read_csv("../classification/breast_cancer.csv")
# df.drop(["Unnamed: 32"], axis=1, inplace=True)
# print(df)

# import pandas as pd
# dataframe = pd.read_csv("../classification/breast_cancer.csv")
# if (dataframe['diagnosis'].dtypes) == "object":
#     print(dataframe["diagnosis"].unique())
# for i in range(len(dataframe.columns)):
#     missing_data = dataframe[dataframe.columns[i]].isna().sum()
#     perc = missing_data / len(dataframe) * 100
#     print('>%d,  missing entries: %d, percentage %.2f' % (i, missing_data, perc))


# import numpy as np
  
# # Importing the SimpleImputer class
# from sklearn.impute import SimpleImputer
  
# # Imputer object using the mean strategy and 
# # missing_values type for imputation
# imputer = SimpleImputer(missing_values = np.nan, 
#                         strategy ='mean')
  
# data = [[12, np.nan, 34], [10, 32, np.nan], 
#         [np.nan, 11, 20]]
  
# print("Original Data : \n", data)
# # Fitting the data to the imputer object
# imputer = imputer.fit(data)
  
# # Imputing the data     
# data = imputer.transform(data)
  
# print("Imputed Data : \n", data)
