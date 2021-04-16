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
import pickle, joblib
from threading import Thread
from IPython.display import display
import model_dispatch
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
    def __init__(self, path, label, fold=False, test_size=0.25, save_model_type="joblib"):
        super(AutoDeepClassification, self).__init__()
        self.path = path
        self.label = label
        self.fold = fold
        self.seed = 42
        self.test_size = test_size
        self.save_model_type = save_model_type
        
    def display_option(self, df):
        return display(df)
    
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
        self.display_option(df)
        return df

    def data_clean_extra_column(self, df):
        '''
        if you want to remove extra columns
        '''
        for col in df.columns:
            col_name = input(f"[INPUT] ENTER THE COLUMN NAME YOU WANT TO REMOVE / TYPE 'quit' : ")
            if col_name in df.columns:
                df.drop([col], axis=1, inplace=True)
            elif col_name == "quit":
                break
            elif col_name not in df.columns:
                print(f"[WRONG] INPUT CORRECT COLUMN NAME")

    def scale_data(self, x):
        standardization = preprocessing.StandardScaler().fit(x)
        df_std = standardization.transform(x)
        scaled_datframe = pd.DataFrame(df_std, index=x.index, columns=x.columns)
        self.display_option(scaled_datframe)
        return scaled_datframe


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

        if len(unique_label) <= 2:
            '''
                Create a right and suggestion type system
                if you choose one-hot:
                    will say "right"
                else: will say "suggested line"
            '''
            if data_input == "one-hot":
                print(f"[INFO] THATS RIGHT")
            else:
                print(f"[INFO] Suggest to apply One-Hot Encoding\n\tBecause, \n\tThe categorical feature is not ordinal\n\tThe number of categorical features is less so one-hot encoding can be effectively applied\n") 
        elif len(unique_label) > 2:
            if data_input != "one-hot":
                print(f"[INFO] THATS RIGHT")
            else:
                print(f"[INFO] Suggest to apply Label Encoding\n\tBecause, \n\tThe categorical feature is ordinal\n\tThe number of categories is quite large as one-hot encoding can lead to high memory consumption\n")

        if len(unique_label) <= 3 and df[self.label].dtypes == "object" and data_input == "one-hot":
            dummy2 = pd.get_dummies(df[self.label])
            df = pd.concat([dummy2, df], axis=1).drop(self.label, axis=1)
        else:    
            labelencode = preprocessing.LabelEncoder()
            df[self.label] = labelencode.fit_transform(df[self.label]) 
            inv_label = labelencode.inverse_transform(df[self.label].unique())
        return df, inv_label

    def train_val_data(self, df):
        X = df.drop([self.label], axis=1)
        y = df[self.label]
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, random_state=self.seed, test_size=self.test_size
        )
        print(f"[INFO] DATA IS SPLITED\n\tTRAIN RATIO : {1-self.test_size}\n\tTEST RATIO : {self.test_size}")
        return X_train, X_test, y_train, y_test

    def model_(self, X_train, X_test, y_train, y_test):
        scores = []
        for model_name, param in model_dispatch.model_params.items():
            clf = model_selection.GridSearchCV(param["model"], param["params"], cv=5, return_train_score=False)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = self.model_evaluation(y_pred, y_test)
            if self.save_model_type == "pkl":
                self.save_model_for_gridsearch(clf, f"{os.path.join(os.path.dirname(os.getcwd()), 'output')}/{model_name}.pkl")
            else:
                self.save_model_for_gridsearch(clf, f"{os.path.join(os.path.dirname(os.getcwd()), 'output')}/{model_name}.joblib")
            scores.append({
                "model": model_name,
                "best_score": clf.best_score_,
                "best_params": clf.best_params_,
                "test_accuracy": accuracy,
                "save_path": f"{os.path.join(os.path.dirname(os.getcwd()), 'output')}/{model_name}.{self.save_model_type}"
            })
        return pd.DataFrame(scores)

    def prediction(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred

    def save_model_for_gridsearch(self, model, Filename):
        if not os.path.exists(os.path.join(os.path.dirname(os.getcwd()), "output")):
            os.makedirs(os.path.join(os.path.dirname(os.getcwd()), "output"))
        if self.fold == False:
            if self.save_model_type == "pickle":
                with open(Filename, 'wb') as file:  
                    pickle.dump(model, file)
            else:
                with open(Filename, 'wb') as file:  
                    joblib.dump(model, file)


    def model_evaluation(self, predicted, y_test_value):
        accuracy_check = accuracy_score(predicted, y_test_value)
        return accuracy_check

    def load_datafile(self, filename, df):
        new_data = []
        all_file = os.listdir(os.path.join(os.path.dirname(os.getcwd()), "output"))
        file_type = all_file[0].split('.')[1]
        

        if file_type == "joblib":
            model = joblib.load(filename)
        else:
            model = pickle.load(filename)
        for data in df.columns:
            data = float(input(f"[INPUT] COLUMN NAME : {data} ENTER VALUE : "))
            new_data.append(data)
        test_data = np.array(new_data).reshape(1, -1)
        standardization_test = preprocessing.StandardScaler().fit(test_data)
        std_test = standardization_test.transform(test_data)
        pred = model.predict([new_data])
        return pred

    def train(self):
        '''
        load_data_folder: give the path of the folder
        csv : return a csv file
        '''
        for encoding in set(aliases.values()):
            try:
                dataframe = pd.read_csv(self.path, encoding=encoding)
                print(f"[SUCCESS] Dataset Loaded Successfully")
                self.display_option(dataframe)
                label_type = dataframe[self.label][0]
                self.data_clean(df=dataframe)   # Data cleaning null columns
                if self.fold == True: 
                    df = self.create_folds(df=dataframe)
                else: 
                    df = self.with_out_kfold(df=dataframe)

                if (isinstance(label_type, np.int64)):
                    print("True")
                    std_df = df.drop([self.label], axis=1)
                else:
                    data_input = input(f"[INPUT] ENTER THE NAME OF ENCODER BECAUSE LABEL DTYPE IS OBJECT : ")
                    split_df, inv_label = self.split_data(df, data_input)
                    X = split_df.drop([self.label], axis=1)
                    print("False")
                    std_df = self.scale_data(X)
                new_scaled_data = pd.concat([df[self.label], std_df], axis=1)
                X_train, X_test, y_train, y_test = self.train_val_data(new_scaled_data)
                scores = self.model_(X_train, X_test, y_train, y_test)
                self.display_option(scores.sort_values(by="best_score", ascending=False, ignore_index=True))
                filename = input(f"[INPUT] ENTER THE PATH : ")
                if os.path.exists(filename): print(f"[INFO] LOADED SUCCESSFULLY")
                else: print(f"[ERROR] FILE NOT FOUND")
                pred = self.load_datafile(filename, new_scaled_data.drop([self.label], axis=1))
                return f"[RESULT] predicted label : {inv_label[pred]}"
            except Exception as e:
                print(f"[ERROR] {e}")
                break

        



# df = AutoDeepClassification(path="../classification/iris.csv", label="Species", fold=False)
# a = df.train()
# print(a)

