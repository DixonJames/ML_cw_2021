from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import urllib.request
import tarfile
import os
import math
import re
from dateutil import parser
from datetime import datetime


DOWNLOAD_ROOT = "https://github.com/beoutbreakprepared/nCoV2019/raw/master/latest_data/latestdata.tar.gz"
DATA_PATH = os.path.join("datasets", "COVID")
COVID_URL = DOWNLOAD_ROOT + "datasets/COVID/COVID.tgz"

def fetch_covid_data(covid_url=DOWNLOAD_ROOT, data_path=DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    tgz_path = os.path.join(data_path, "COVID.tgz")
    urllib.request.urlretrieve(covid_url, tgz_path)
    covid_tgz = tarfile.open(tgz_path)
    covid_tgz.extractall(path=data_path)
    covid_tgz.close()

def storeDataframe(df, path, name = "stored_df", zipped = False):
    if zipped == False:
        df.to_csv(name, sep=',', encoding='utf-8')
    else:
        comp_options = {"method":"zip", "archive_name":f"{name}"}
        df.to_csv(f"{name}.zip", index=False, compression=comp_options, path_or_buf=path)

def load_data(housing_path=DATA_PATH):
    csv_path = os.path.join(housing_path, "latestdata.csv")
    return pd.read_csv(csv_path)

def dropcolls(df, coll = None):
    if coll != None:
        df.drop([coll], axis = 1)
    else:
        useless_cols = ["travel_history_dates", "ID","symptoms","chronic_disease", "lives_in_Wuhan","reported_market_exposure","additional_information","source", "geo_resolution","travel_history_location","sequence_available","notes_for_discussion","admin3","admin2","admin1","country_new","admin_id","data_moderator_initials"]
        df.drop(useless_cols, axis=1, inplace = True)

def nanCheck(val):
    try:
        return math.isnan(val)
    except:
        return False

def numCheck(val):
    try:
        float(val)
        return True
    except:
        return False


class DateCleaning:
    def cleanDate(self, date):

        if "-" in date:
            dateA = parser.parse(date.split("-")[0], dayfirst=True)
            dateB = parser.parse(date.split("-")[1], dayfirst=True)
            try:
                return dateA + (dateB - dateA)/2
            except (ValueError, TypeError) as e:
                return False
                print(e, date)

        try:
            return parser.parse(date, dayfirst=True)
        except (ValueError, TypeError) as e:
            return False
            print(e, date)

    def dayComp(self, d1):
        return d1.year * 365 + d1.month * 30 + d1.day * 1

    def convertDates(self, collumb):
        age_labels = collumb.unique()
        dates = []

        transDict = {'nan': 'nan'}
        for date in age_labels:
            try:
                if not (nanCheck(date)):
                    if self.cleanDate(date) != False:
                        #collumb = collumb.replace(date, cleanDate(date))
                        clean = self.cleanDate(date)
                        transDict[date] = clean
                        dates.append(clean)
            except:
                transDict[date] = np.nan
                dates.append(np.nan)


        zeroday = datetime(2019, 12, 1)

        collumb = collumb.replace([date for date in age_labels if not (nanCheck(date))], [(transDict[date] - zeroday).days for date in age_labels if not (nanCheck(date))])

        return collumb

    def tryFloat(self, test):
        try:
            float(test)
            return True
        except:
            return False



    def medaanAges(self, collumb):
        age_labels = whole_DF["age"].unique()

        transdict = {}
        for i in range(1000):
            transdict[i] = i

        for age in age_labels:
            changed = False
            value = str(age)
            if not (nanCheck(age)):


                if self.tryFloat(value):
                    if float(value) > 1.0:
                        transdict[age] = float(value)

                        changed = True
                    else:
                        transdict[age] = float(1)
                        changed = True

                if re.search("-", value) != None and changed == False:
                    parts = value.split("-")
                    if parts[1] != "":
                        if self.tryFloat(parts[0]) and self.tryFloat(parts[1]):
                            transdict[age] = float((float(parts[1]) + float(parts[0])) / 2)
                            changed = True

                if (re.search("months", value) != None or re.search("month", value) != None or re.search("weeks", value) != None) and changed == False:
                    transdict[age] = float(1)
                    changed = True

                if re.search("\+", value) != None and changed == False:
                    transdict[age] = float(float(re.sub("\+", "", value)))

                    changed = True

                if re.search("\-", value) != None and changed == False:
                    transdict[age] = float(float(re.sub("\-", "", value)))
                    changed = True

        collumb = collumb.replace(transdict)

        return collumb


class DataManipulation:
    @staticmethod
    def checkNan(value):
        if nanCheck(value) == True or numCheck(value):
            return value
        return np.nan

    @staticmethod
    def scaleData(data):
        #scales data to between 0 and 1
        from sklearn.preprocessing import normalize, MinMaxScaler

        min_max_scaler = MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(pd.DataFrame(data))

        return pd.DataFrame(data_scaled)

    @staticmethod
    def binData(df):
        labels = df.unique()
        labels = [l for l in labels if not (nanCheck(l))]


        transDict = {'nan': 'nan'}
        for category in labels:
            if not (nanCheck(category)):
                transDict[category] = labels.index(category)

        df = df.replace([date for date in labels if not (nanCheck(date))], [transDict[date] for date in labels if not (nanCheck(date))])

        return df

    @staticmethod
    def cleanOutcome(df):
        live_labels = ["Recovered","recovered","Alive","discharge","stable","discharged", "Hospitalized"]
        die_labels = ["Deceased","death","Dead","Died","Death","dead"]
        uncirtain_labels = []

        #james toursends dict
        outcome_dict = {
            "Alive": 0,
            "Critical condition": 0,
            "Dead": 1,
            "Death": 1,
            "Deceased": 1,
            "Died": 1,
            "Discharged": 0,
            "Discharged from hospital": 0,
            "Hospitalised": 0,
            "Migrated": 0,
            "Migrated_Other": 0,
            "Receiving Treatment": 0,
            "Recovered": 0,
            "Stable": 0,
            "Symptoms only improved with cough. Currently hospitalized for follow-up.": 0,
            "Under treatment": 0,
            "critical condition": 0,
            "critical condition, intubated as of 14.02.2020": 1,
            "dead": 1,
            "death": 1,
            "died": 1,
            "discharge": 0,
            "discharged": 0,
            "https://www.mspbs.gov.py/covid-19.php": 0,
            "not hospitalised": 0,
            "recovered": 0,
            "recovering at home 03.03.2020": 0,
            "released from quarantine": 0,
            "severe": 0,
            "severe illness": 0,
            "stable": 0,
            "stable condition": 0,
            "treated in an intensive care unit (14.02.2020)": 0,
            "unstable": 1,
            'nan': np.nan,
            np.nan: np.nan
        }

        labels = df.unique()

        transDict = {'nan': np.nan, np.nan: np.nan}
        for l in labels:
            transDict[l] = 1


        df = df.replace(transDict)

        return df

    @staticmethod
    def OneHotEncoding(frame, cols):
        from sklearn.preprocessing import OneHotEncoder

        OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)


        en_valid_cols = pd.DataFrame(OHE.fit_transform(frame[cols]))

        #repalce indexes

        en_valid_cols.index = frame.index

        #remove old categorical columbs

        num_X_valid = frame.drop(cols, axis=1)

        #stitch it all together

        en_valid = pd.concat([num_X_valid, en_valid_cols], axis=1)

        return en_valid

    @staticmethod
    def easyOneHot(df, target_col):
        encodedCols = pd.get_dummies(df[target_col])

        for col in list(encodedCols):
            whole_DF[target_col + "_" + str(col)] = encodedCols[col]
        whole_DF.drop(columns=[target_col], inplace=True)

    @staticmethod
    def timeSpanUnitilHospitalisation(row):
        if not(nanCheck(row["date_admission_hospital"])) and not(nanCheck(row["date_onset_symptoms"])):
            return abs(row["date_admission_hospital"] - row["date_onset_symptoms"])
        return np.nan

    @staticmethod
    def confirmationUnitilHospitalisation(row):
        if not(nanCheck(row["date_admission_hospital"])) and not(nanCheck(row["date_confirmation"])):
            return row["date_confirmation"] - row["date_admission_hospital"]
        return np.nan

    @staticmethod
    def symptomsUnitilconfirmation(row):
        if not(nanCheck(row["date_onset_symptoms"])) and not(nanCheck(row["date_confirmation"])):
            return row["date_confirmation"] - row["date_onset_symptoms"]
        return np.nan





class ImputingData:
    def __init__(self, df):
        self.df = df
        self.city = df["city"]
        self.country = df["country"]
        self.province = df["province"]
        self.latitude = df["latitude"]
        self.longitude = df["longitude"]


    def imputeHospitalStayDates(self):
        workspace = pd.DataFrame({"date_admission_hospital" : self.df["date_admission_hospital"], "date_death_or_discharge" : self.df["date_death_or_discharge"]})
        workspace = workspace.dropna()
        workspace["stay_time"] = workspace["date_death_or_discharge"] - workspace["date_admission_hospital"]
        avg_stay = workspace["stay_time"].mean()

        impute_workspace = pd.DataFrame({"date_admission_hospital": self.df["date_admission_hospital"],
                                  "date_death_or_discharge": self.df["date_death_or_discharge"]})

        def dateChange(startdate, daychange):
            return startdate.loc[~startdate.isnull()] + daychange

        impute_workspace.loc[impute_workspace['date_death_or_discharge'].isnull(), 'date_death_or_discharge'] = dateChange(impute_workspace['date_admission_hospital'], avg_stay)
        impute_workspace.loc[impute_workspace['date_admission_hospital'].isnull(), 'date_admission_hospital'] = dateChange(impute_workspace['date_death_or_discharge'], -1 * avg_stay)


        return impute_workspace['date_death_or_discharge'], impute_workspace['date_admission_hospital']


    def fillLLVals(self):
        return self.latitude, self.longitude

    def getLL(self):
        """
        imputes the Latitude and longitude values by the median lat and long in the priority order of:
        1. city
        2. province
        3. country
        :return: lat and long cols with less np.nan values
        """
        df_secton = {"country": self.country,
                     "city": self.city,
                     "province": self.province,
                     "latitude": self.latitude,
                     "longitude": self.longitude}
        df_secton = pd.DataFrame(df_secton, columns=["country","city","province","latitude","longitude"])

        transDf = self.MeanLLFromCol("city")
        citytoLLDict = transDf.set_index("city").to_dict()

        countryDf = self.MeanLLFromCol("country")
        countryDict = countryDf.set_index("country").to_dict()

        provDf = self.MeanLLFromCol("province")
        provDict = provDf.set_index("province").to_dict()

        df_secton['latitude'].fillna(df_secton['city'].astype(str).map(citytoLLDict['latitude']), inplace = True)
        df_secton['latitude'].fillna(df_secton['city'].astype(str).map(citytoLLDict['latitude']), inplace = True)

        df_secton['latitude'].fillna(df_secton['province'].astype(str).map(provDict['latitude']), inplace=True)
        df_secton['latitude'].fillna(df_secton['province'].astype(str).map(provDict['longitude']), inplace=True)

        df_secton['latitude'].fillna(df_secton['country'].astype(str).map(countryDict['latitude']), inplace = True)
        df_secton['latitude'].fillna(df_secton['country'].astype(str).map(countryDict['longitude']), inplace = True)

        self.country = df_secton["country"]
        self.latitude = df_secton["latitude"]
        self.longitude = df_secton["longitude"]


    def MeanLLFromCol(self, col):
        colToLong = self.df.groupby(col, as_index=False)['longitude'].mean()
        colToLat = self.df.groupby(col, as_index=False)['latitude'].mean()
        colToLong.insert(2, "latitude", colToLat['latitude'], True)

        return colToLong



if __name__ == '__main__':
    #fetch_housing_data()

    whole_DF = load_data()
    dropcolls(whole_DF)
    whole_DF = whole_DF[whole_DF.date_onset_symptoms.notnull()]



    # cleaning ages working
    whole_DF["age"] = DateCleaning().medaanAges(whole_DF["age"])
    whole_DF['age'] = whole_DF['age'].fillna(whole_DF.groupby('country').age.transform('median'))





    #imputing lat and long from other geo data (city and country)
    imputer = ImputingData(whole_DF)
    imputer.getLL()
    whole_DF["latitude"], whole_DF["longitude"] = imputer.fillLLVals()
    """
    whole_DF['latitude'] = whole_DF['latitude'].fillna(whole_DF.groupby('city').latitude.transform('median'))
    whole_DF['longitude'] = whole_DF['longitude'].fillna(whole_DF.groupby('city').longitude.transform('median'))

    whole_DF['latitude'] = whole_DF['latitude'].fillna(whole_DF.groupby('province').latitude.transform('median'))
    whole_DF['longitude'] = whole_DF['longitude'].fillna(whole_DF.groupby('province').longitude.transform('median'))

    whole_DF['latitude'] = whole_DF['latitude'].fillna(whole_DF.groupby('country').latitude.transform('median'))
    whole_DF['longitude'] = whole_DF['longitude'].fillna(whole_DF.groupby('country').longitude.transform('median'))
    """

    whole_DF.drop(columns=["province", "city", "location"], inplace=True)



    #turn sex binary
    DataManipulation.easyOneHot(whole_DF, "sex")
    DataManipulation.easyOneHot(whole_DF, "chronic_disease_binary")
    DataManipulation.easyOneHot(whole_DF, "travel_history_binary")

    # """


    #cleaning dates working
    datecols = ["date_onset_symptoms", "date_admission_hospital", "date_confirmation", "date_death_or_discharge"]
    for atrb in datecols:
        whole_DF[atrb] = DateCleaning().convertDates(whole_DF[atrb])

    #imputes these two dates from each other depending on the mean differnace bwtween then in the records crountry
    whole_DF['date_death_or_discharge'], whole_DF['date_admission_hospital'] = imputer.imputeHospitalStayDates()



    #adds  new cols to the data
    #time_symptoms_until_hospital is the label

    #label
    whole_DF['time_symptoms_until_hospital'] = whole_DF.apply(lambda row: DataManipulation.timeSpanUnitilHospitalisation(row), axis=1)
    #extras
    whole_DF['time_hospital_until_confirmation'] = whole_DF.apply(lambda row: DataManipulation.confirmationUnitilHospitalisation(row), axis=1)
    whole_DF['time_symptoms_until_hospital'] = whole_DF.apply(lambda row: DataManipulation.symptomsUnitilconfirmation(row), axis=1)




    #turns outcome into live/die bianry
    binary_cols = ["outcome"]
    for atrb in binary_cols:
        whole_DF[atrb] = DataManipulation.cleanOutcome(whole_DF[atrb])


    #does one hot encoding for each country
    tmp = ["country"]
    for atrb in tmp:
        whole_DF[atrb] = DataManipulation.easyOneHot(whole_DF, "country")

    #"""


    whole_DF = whole_DF[whole_DF.date_admission_hospital.notnull()]
    whole_DF = whole_DF[whole_DF.date_confirmation.notnull()]
    whole_DF = whole_DF[whole_DF.age.notnull()]
    whole_DF = whole_DF[whole_DF.latitude.notnull()]
    whole_DF = whole_DF[whole_DF.longitude.notnull()]
    whole_DF.reset_index(inplace=True)

    #noramlising data
    noramlising_colls = ["latitude", "longitude", "date_death_or_discharge", "date_admission_hospital", "date_onset_symptoms",  "date_confirmation", "time_symptoms_until_hospital", "time_hospital_until_confirmation", "time_symptoms_until_hospital", "age"]
    for atrb in noramlising_colls:
        whole_DF[atrb] = DataManipulation.scaleData(whole_DF[atrb])

    nanFilter = DataManipulation().checkNan
    whole_DF.applymap(nanFilter, na_action='ignore')
    storeDataframe(whole_DF, os.getcwd(), name="cleaned_df.csv")
    #120726

