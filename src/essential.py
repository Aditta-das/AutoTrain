import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble

from sklearn.metrics import accuracy_score

def visualize_dataframe(data):
    return pd.DataFrame(data)


def split_data_according_to_fold(fold, df):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    test_df = df[df.kfold == fold].reset_index(drop=True)
    return train_df, test_df

def scalization(features):
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(features)
    return X_df

def split_dataset(train_part, target):
    X = train_part.drop(target, axis=1)
    y = train_part[target]
    seed = 42
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=seed, test_size=0.3)
    # X_train, _, y_train, _ = model_selection.train_test_split(X, y, random_state=seed)
    # _, X_test, _, y_test = model_selection.train_test_split(X, y, random_state=seed)
    return X_train, X_test, y_train, y_test

def model(X_train, X_test, y_train, y_test):
    scores = []
    # clf = linear_model.LogisticRegression()
    # predict = clf.fit(X_train, y_train).predict(X_test)
    # accuracy = accuracy_score(y_test, predict)
    # return accuracy
    model_params = {
        "svm": {
            "model": svm.SVC(gamma='auto'),
            "params": {
                "C": [1, 10, 20],
                "kernel": ["rbf", "linear", "poly", "sigmoid"]
            }
        },
        "RandomForest": {
            "model": ensemble.RandomForestClassifier(),
            "params": {
                "n_estimators": [100, 150, 200],
                "criterion": ["gini", "entropy"],
                "max_depth": [5, 10, 15],
                "max_features": ["auto", "sqrt", "log2"],
            }
        },
        "LogisticRegression": {
            "model": linear_model.LogisticRegression(),
            "params": {
                "C": [1, 10, 20],
                "penalty": ["l1", "l2", "elasticnet", "none"],
                "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                "multi_class": ["auto", "ovr", "multinomial"],
            }
        },
        "KNN": {
            "model": neighbors.KNeighborsClassifier(),
            "params": {
                "n_neighbors": [5, 10, 15],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            }
        },
        "ExtraTree": {
            "model": ensemble.ExtraTreesClassifier(),
            "params": {
                "n_estimators" : [100, 150, 200],
                "criterion": ["gini", "entropy"],
                "max_features":["auto", "sqrt", "log2"],
            }
        },
        "DecisionTree": {
            "model": tree.DecisionTreeClassifier(),
            "params": {
                "criterion": ["gini", "entropy"]
            }
        }
    }
    for model_name, param in model_params.items():
        clf = model_selection.GridSearchCV(param["model"], param["params"], return_train_score=False)
        clf.fit(X_train, y_train)
        scores.append({
            "model": model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_,   
        })
    return scores
        

