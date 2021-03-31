import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import os

DATA_PATH = os.path.join("datasets", "COVID", "cleaned_df.csv")


def load_data(housing_path=DATA_PATH):
    csv_path = os.path.join(housing_path)
    df = pd.read_csv(csv_path)
    df.drop(columns=["index", "Unnamed: 0"], inplace=True)
    df.astype('float64')

    return df


class DataSplit:
    def __init__(self, whole_df, split):
        self.data = whole_df.drop(columns=["time_symptoms_until_hospital"])
        self.target = whole_df["time_symptoms_until_hospital"]

        self.split = split

    def predictionAccuracey(self, model, k=None):
        from sklearn.model_selection import cross_val_score
        if k == None:
            k = self.split

        X = self.data
        Y = self.target
        folder = KFold(n_splits=k, random_state=None)

        return cross_val_score(model, X, Y, cv=folder)

    def kFoldGrouping(self, k=5):
        X = self.data
        Y = self.target
        folder = KFold(n_splits=k, random_state=None)

        for train_i, test_i in folder.split(X):
            yield X.iloc[train_i, :], X.iloc[test_i, :], Y[train_i], Y[test_i]


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

        return grid.best_score_, best_res

    def randomHyperP(self, perams, model, X, Y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RepeatedKFold

        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define search space

        search = GridSearchCV(model, perams, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
        print(search)
        # execute search
        result = search.fit(X, Y)
        # summarize result

        return result.best_params_

    def graph_ts(self, values, DS):
        from sklearn.metrics import mean_squared_error

        results_train = []
        results_test = []

        for k in values:

            datasetCreation = DataSplit(whole_df=DS, split=5)
            dataSetGen = datasetCreation.kFoldGrouping()

            model = RandomForestRegressor(min_impurity_decrease = k)


            error_test = []
            error_train = []
            for sets in range(4):
                print(k,sets)
                X_train, x_test, Y_train, y_test = next(dataSetGen)
                trained_m = model.fit(X_train, Y_train)

                predictions_test = trained_m.predict(x_test)
                predictions_train = trained_m.predict(X_train)

                error_test.append(mean_squared_error(y_test, predictions_test))
                error_train.append(mean_squared_error(Y_train, predictions_train))

            results_train.append((sum(error_test) / len(error_test)))
            results_test.append((sum(error_train) / len(error_train)))

        #return results_test, results_train

        results_test = pd.DataFrame(results_test)
        results_test['id'] = [i for i in range(len(results_test))]
        results_test['val'] = results_test[0]
        results_test['source'] = ["test" for i in range(len(results_test))]

        results_train = pd.DataFrame(results_train)
        results_train['id'] = [i for i in range(len(results_train))]
        results_train['val'] = results_train[0]
        results_train['source'] = ["train" for i in range(len(results_train))]

        both_lines = results_train.append(results_test)
        both_lines.drop(columns=[0], axis=1, inplace=True)

        both_lines_wide = both_lines.pivot("id", "val", "source")

        sns.lineplot(x="id", y='val', data=both_lines, hue="source")


def main():
    # get data and model
    cleaned_df = load_data()
    #model = KNeighborsRegressor()
    model = RandomForestRegressor()

    # gening d sets
    dataset_creation = DataSplit(whole_df=cleaned_df, split=5)
    data_set_gen = dataset_creation.kFoldGrouping()
    x_train, x_test, Y_train, y_test = next(data_set_gen)

    PeramTuning(cleaned_df).graph_ts([float(i)/10 for i in range(1,50)] ,cleaned_df)


    #best K graph
    #PeramTuning(cleaned_df).graph_ks(10, cleaned_df)


    # settting up hyperperams to tune
    R_space = dict()

    #R_space["n_estimators"] = [40]
    R_space["criterion"] = ["mse"]
    #R_space["min_samples_split"] = [2]
    #R_space["min_samples_leaf"] = [1]





    op_perams = PeramTuning(cleaned_df).randomHyperP(R_space, model, x_train, Y_train)
    print(op_perams)


main()
