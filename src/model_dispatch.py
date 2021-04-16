from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
# models = [SVC, RandomForestClassifier, LogisticRegression, ExtraTreeClassifier,  DecisionTreeClassifier]
# params = []

# for i, model in enumerate(models):
#     model_name = model()
#     a = model_name.get_params()
#     params.append({
#         f"model_{i}": model(),
#         f"params_{i}" : a
#     })

# for p in params:
#     params[0]["params_0"].update(
#         {'C': [x for x in range(10, 100, 10)], 'break_ties': False, 'cache_size': 200, 'class_weight': None, 
#         'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 
#         'gamma': ['scale', 'auto'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'max_iter': -1, 'probability': False, 
#         'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
#     )
#     params[1]["params_1"].update(
#         {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': ['balanced', 'balanced_subsample', None], 'criterion': ['gini', 'entropy'], 
#         'max_depth': None, 'max_features': ['auto', 'sqrt', 'log2'], 'max_leaf_nodes': None, 'max_samples': None, 
#         'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': [x for x in range(10)], 
#         'min_samples_split': [x for x in range(2, 12, 2)], 'min_weight_fraction_leaf': 0.0, 'n_estimators': [x for x in range(100, 500, 100)], 
#         'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
#     )

# print(params)


#     for key, val in a.items():
#         print(key)
#         params.append(key)
# print(params)

model_params = {
    "svm": {
        "model": SVC(gamma="auto"),
        "params":{
            "C": [x for x in range(10, 100, 10)],
            "kernel": ["rbf", "linear"]
        }
    },
    "KNearest": {
        "model": KNeighborsClassifier(),
        "params": {
            "weights": ["uniform", "distance"],
            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
            "leaf_size": [30, 35, 40],
            "n_neighbors": [5, 10, 15],
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [x for x in range(5, 15, 5)]
        }
    },
    "Logistic Regression":{
        "model": LogisticRegression(solver="liblinear", multi_class="auto"),
        "params": {
            "C": [x for x in range(1, 10)]
        }
    },
    "ExtraTreeClassifier": {
        "model": ExtraTreeClassifier(),
        "params": {
            "criterion": ['gini', 'entropy'],
            "max_features": ['auto', 'sqrt', 'log2'],
            "min_samples_split": [x for x in range(2, 10, 2)],
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [x for x in range(2, 10, 2)],
            "min_samples_split": [x for x in range(2, 10, 2)],
            "max_features": ['auto', 'sqrt', 'log2', None],   
        }
    },
    "GradingBoosting": {
        "model": GradientBoostingClassifier(),
        "params": {
            "loss": ['deviance', 'exponential'],
            "learning_rate": [0.1],
            "n_estimators":[x for x in range(10, 40, 10)],
            "criterion": ['friedman_mse', 'mse', 'mae'],
            "min_samples_split": [x for x in range(2, 8, 2)],
            "max_features": ['auto', 'sqrt', 'log2', None],
        }
    },
    "XGBoost": {
        "model": XGBClassifier(),
        "params": {
            "objective": ['binary:logistic', 'multi:softmax', 'multi:softprob'],
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15]
        }
    }
}