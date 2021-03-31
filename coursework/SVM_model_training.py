import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
DATA_PATH = os.path.join("datasets", "COVID", "datasets/COVID/cleaned_df.csv")

def load_data(housing_path=DATA_PATH):
    csv_path = os.path.join(housing_path)
    return pd.read_csv(csv_path)


class DataSplit:
    def __init__(self, whole_df, split):
        self.data = whole_df.drop(columns=["time_symptoms_until_hospital"])
        self.target = whole_df["time_symptoms_until_hospital"]

        self.split = split

    def predictionAccuracey(self, model, k = None):
        from sklearn.model_selection import cross_val_score
        if k == None:
            k = self.split


        X = self.data
        Y = self.target
        folder = KFold(n_splits=k, random_state=None)

        return cross_val_score(model, X, Y, cv=folder)

    def kFoldGrouping(self, k = 5):
        X = self.data
        Y = self.target
        folder = KFold(n_splits=k, random_state=None)

        for train_i, test_i in folder.split(X):
            yield X.iloc[train_i,:], X.iloc[test_i,:], Y[train_i] , Y[test_i]

class PeramTuning:
    def __init__(self, whole_df):
        self.data = whole_df.drop(columns=["time_symptoms_until_hospital"])
        self.target = whole_df["time_symptoms_until_hospital"]


    def fineTuneHyperP(self, p_grid, model):
        from sklearn.model_selection import GridSearchCV

        grid = GridSearchCV(estimator=model, param_grid=p_grid)
        grid.fit(self.data, self.target)

        best_res = dict()
        for peram in list(p_grid):
            best_res[peram] = grid.best_estimator_.peram


        return grid.best_score_,best_res

    def randomHyperP(self, model, X, Y):
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import RepeatedKFold

        from scipy.stats import uniform as sp_rand
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define search space
        space = dict()
        #space['kernel'] = ['linear', 'rbf']
        space['C'] = range(1, 10) #6
        space['degree'] = [1,2,3]
        space['gamma'] = ["auto"]
        # define search
        search = RandomizedSearchCV(model, space, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-4, cv=cv,
                                    random_state=1)
        # execute search
        result = search.fit(X, Y)
        # summarize result


        return result.best_params_



class SVC:
    def __init__(self):
        pass



def main():
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_score
    cleaned_df = load_data()
    model = SVR()

    datasetCreation = DataSplit(whole_df=cleaned_df, split=5)
    dataSetGen = datasetCreation.kFoldGrouping()
    X_train, x_test, Y_train, y_test = next(dataSetGen)

    #cross_val_score(model, X_train, x_test, cv=5, scoring='recall_macro')

    p_workspace = PeramTuning(cleaned_df)
    perams = p_workspace.randomHyperP(model, X_train, Y_train)

    models = {}
    for X_train, x_test, Y_train, y_test in  next(dataSetGen):
        model = SVR(random_state=0, gamma= perams['gamma'], degree= perams["degree"], C= perams["C"])
        model.fit(X_train, Y_train)



    """
    regr = SVR(C=1.0, epsilon=0.2)
    regr.fit(X_train, Y_train)
    """






    p_workspace = PeramTuning(cleaned_df)
    vals = ['C', 'epsilon']

    print(p_workspace.randomHyperP(model, X_train, Y_train))

if __name__ == '__main__':
    main()