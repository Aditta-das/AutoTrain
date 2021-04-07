import pandas as pd 
import numpy as np 
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score

import essential

import warnings
warnings.filterwarnings("ignore")
print(f"[WARNING] All Warnings are ignored")

class AutoDeep:
    def __init__(self, path, target_path, fold_no, target, size=0.3):
        super(AutoDeep, self).__init__()
        self.path = path
        self.target_path = target_path
        self.fold_no = fold_no
        self.target = target
        self.size = size
        self.scores = []    

    def load_data(self, path):
        '''
        load_data_folder: give the path of the folder
        csv : return a csv file
        '''
        dataset = pd.read_csv(path)
        print(f"[INFO] Data Load Complete")
        return dataset
    
    def __len__(self):
        dataset = self.load_data(self.path)
        return f"[INFO] Row {dataset.shape[0]} || Column {dataset.shape[1]}"

    def preprocess_data(self):
        df = self.load_data(self.path)
        print(self.__len__())
        df["kfold"] = -1
        df = df.sample(frac=1).reset_index(drop=True)
        y = df[self.target].values
        kf = model_selection.StratifiedKFold(n_splits=self.fold_no)
        for fold, (tr, val) in enumerate(kf.split(X = df, y=y)):
            df.loc[val, "kfold"] = fold
        df.to_csv(self.target_path, index=False)
        print(f"[INFO] Created New Fold Column")
        # preprocessing
        new_df = self.load_data(self.target_path)
        corr_mat = new_df.corr()
        mk_data = corr_mat[self.target].sort_values(ascending=False)
        correlation_matrix = essential.visualize_dataframe(mk_data)

        features = new_df.drop([self.target, "kfold"], axis=1)
        scale = essential.scalization(features)
        scaled_df = pd.DataFrame(scale, columns=features.columns)
        scaled_df = pd.concat([scaled_df, new_df[[self.target, "kfold"]]], axis=1)

        return correlation_matrix, scaled_df


    def model_apply(self):
        corr, dataset = self.preprocess_data()
        X_train, X_test, y_train, y_test = essential.split_dataset(dataset, target=self.target)
        scores = essential.model(X_train, X_test, y_train, y_test)
        final_result = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
        return corr, final_result

    def run(self):
        corr, final_result = self.model_apply()
        # print(corr)
        return final_result
        # for fold in range(self.fold_no):
        #     accuracy = self.model_apply(fold)
        #     print(accuracy)
        # correlation, train_fold, test_fold = self.preprocess_data(fold_no, target)
        # train_fold, test_fold = essential.split_data_according_to_fold(fold, scaled_df)
        # X_train, X_test, y_train, y_test = essential.split_dataset(train_part=train_fold, target=target, size=size)
        # model = linear_model.LogisticRegression()
        # lr = model.fit(X_train, y_train).predict(X_test)
        # return

        
    def evalution(self):
        pass