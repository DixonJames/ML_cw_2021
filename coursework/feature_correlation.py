import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
DATA_PATH = os.path.join("datasets", "COVID", "cleaned_df.csv")

def load_data(housing_path=DATA_PATH):
    csv_path = os.path.join(housing_path)
    df = pd.read_csv(csv_path)
    df.drop(columns=["index", "Unnamed: 0"], inplace=True)
    df.astype('float64')

    for col in list(df):
        df[col] = df[col].astype('float64')

    return df

def feature_correlecation(df, show = False):

    f_correlation = df.corr(method='pearson')
    f_correlation = np.abs(f_correlation)

    if show:
        sns.heatmap(f_correlation, xticklabels=2, yticklabels=False)
        plt.show()

    return f_correlation

def FeatureColinearityReduction(df):

    corrlation_matrix = feature_correlecation(df.drop("time_symptoms_until_hospital", 1))
    correlation_target = feature_correlecation(df)["time_symptoms_until_hospital"]

    peer_feature_corr = corrlation_matrix.iloc[1:, 1:]
    feature_pairs = (np.round(peer_feature_corr.unstack())).sort_values(ascending=False)

    feature_pairs_sorted = [pair for pair in feature_pairs if ((pair != 1) and (pair >= 0.8))]

    for tupple in feature_pairs_sorted:
        pass





def main():
    whole_df = load_data()
    FeatureColinearityReduction(whole_df)


if __name__ == '__main__':
    main()
