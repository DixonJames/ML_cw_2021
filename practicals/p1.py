# Python ≥3.5 is required
import sys
import sklearn
import pandas as pd
import numpy as np
import os
import tarfile
import urllib.request

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def splitDataSet(ds, state = None):
    import sklearn.model_selection
    if state != None:
        return sklearn.model_selection.train_test_split(ds, test_size=0.2)
    return sklearn.model_selection.train_test_split(ds, test_size=0.2, random_state=state)





def loadData():
    fetch_housing_data()
    housing = load_housing_data()

    #adds an index num
    housing = housing.reset_index()
    print(housing.info())
    return housing



    trainSet, testSet = splitDataSet(housing, 0)

    trainSet.hist(bins=50, figsize=(20, 15))
    save_fig("trainSet_histogram_plots")

    testSet.hist(bins=50, figsize=(20, 15))
    save_fig("testSet_histogram_plots")
    #plt.show()

def fixingMissingData(input_dataframe):
    housing = input_dataframe

    predict_label = "testSet"

    dataframe = housing.drop("median_house_value", axis = 1)
    labels = housing["median_house_value"].copy()

    #finds incomplete rows
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)]


    #dealing with missing data:
    #drop any records with out bedroom atrb
    #sample_incomplete_rows.dropna(subset=["total_bedrooms"])

    #removes tota bedrooms collumb:
    #sample_incomplete_rows.drop("total_bedrooms", axis=1)  # option 2

    #sets any missing totalbedrroms to a median of the present ones
    #median = housing["total_bedrooms"].median()
    #sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)




    #imputing mssing data
    from sklearn.impute import SimpleImputer
    #want medan of each attribute
    imputer = SimpleImputer(strategy="median")

    #ocean_prox is a string that needs removing as median cant be comouted off text
    #housing_num = housing.drop("ocean_proximity", axis=1)
    #or genericaly can do this:
    #removes any non numberic attributes
    numeric_housing = housing.select_dtypes(include=[np.number])

    imputer.fit(numeric_housing)
    #checking if meadian if found for ach attrib
    print(imputer.statistics_ , numeric_housing.median().values)

    #fills in missing data in numeric_housing by its medians
    imputer.transform(numeric_housing)

    #stitckes together filled out dataset
    housing_filled  = pd.DataFrame(imputer.transform(numeric_housing), columns= numeric_housing.columns, index=housing.index)
    # now have no missing values


    return housing_filled

def removingMissingValsCols(x_train, x_test):
    # Get names of columns with missing values
    cols_with_missing = [col for col in x_train.columns
                         if x_train[col].isnull().any()]

    # Drop columns in training and validation data
    reduced_X_train = x_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = x_test.drop(cols_with_missing, axis=1)

    return x_train, x_test

def imputeMissingVals(x_train, x_test):
    from sklearn.impute import SimpleImputer
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(x_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(x_test))

    # Imputation removed column names; put them back
    imputed_X_train.columns = x_train.columns
    imputed_X_valid.columns = x_test.columns

    return x_train, x_test

def imputationExtension(X_train, X_test):
    #adds annother val that indiates an imputed value

    # Make copy to avoid changing original data (when imputing)
    X_train_plus = X_train.copy()
    X_valid_plus = X_test.copy()

    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns

    return imputed_X_train_plus, imputed_X_valid_plus


def removingStringCategories(dataframe):
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import OneHotEncoder, OneHotEncoder

    housing_cat = dataframe[["ocean_proximity"]]

    encodeer = OrdinalEncoder()
    oh_encoder = OneHotEncoder(sparse=False)

    #simple intiger encoding for each ctegory
    housing_cat_encoded = encodeer.fit_transform(housing_cat)

    #turns into one hot ecncoding
    housing_cat_oh_encoded = oh_encoder.fit_transform(housing_cat_encoded)

def scaleData(data):
    #scales data to between 0 and 1
    from mlxtend.preprocessing import minmax_scaling
    return minmax_scaling(data, columns=[0])

def normalisationViaBoxcox(data):
    from scipy import stats
    return stats.boxcox(data)

def labelEncoding(X_train, X_valid, cols):
    from sklearn.preprocessing import LabelEncoder

    # Make copy to avoid changing original data
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()

    # Apply label encoder to each column with categorical data
    label_encoder = LabelEncoder()
    for col in cols:
        label_X_train[col] = label_encoder.fit_transform(X_train[col])
        label_X_valid[col] = label_encoder.transform(X_valid[col])

    return label_X_train, label_X_valid

def OneHotEncoding(X_train, X_valid, cols):
    from sklearn.preprocessing import OneHotEncoder

    OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)
    en_train_cols = pd.DataFrame(OHE.fit_transform(X_train[cols]))
    en_valid_cols = pd.DataFrame(OHE.transform(X_valid[cols]))

    #repalce indexes
    en_train_cols.index = X_train.index
    en_valid_cols.index = X_valid.index

    #remove old categorical columbs
    num_X_train = X_train.drop(cols, axis=1)
    num_X_valid = X_valid.drop(cols, axis=1)

    #stitch it all together
    en_train = pd.concat([num_X_train, en_train_cols], axis=1)
    en_valid = pd.concat([num_X_valid, en_valid_cols], axis=1)

    return en_train, en_valid


def listCategoricalVars(ds):
    s = (ds.dtypes == 'object')
    return list(s[s].index)


if __name__ == '__main__':
    df = loadData()
    patched_df = fixingMissingData(df)

    removingStringCategories(df)




