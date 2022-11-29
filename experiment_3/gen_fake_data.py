import pandas as pd
import numpy as np
from faker import Faker
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

fake = Faker()

# simulate extra rows as if we had a very large dataset
def augment_data(df):
    num_features = len(df.columns)
    a = np.arange(0, num_features)
    aug_df = pd.DataFrame(index=df.index, columns=df.columns, dtype='float64')

    for i in range(0, len(df)):
        AUG_FEATURE_RATIO = 0.5
        AUG_FEATURE_COUNT = np.floor(
            num_features*AUG_FEATURE_RATIO).astype('int16')
        # randomly sample half of columns that will contain random values
        aug_feature_index = np.random.choice(
            num_features, AUG_FEATURE_COUNT, replace=False)
        aug_feature_index.sort()

        # obtain indices for features not in aug_feature_index
        feature_index = np.where(np.logical_not(
            np.in1d(a, aug_feature_index)))[0]

        # first insert real values for features in feature_index
        aug_df.iloc[i, feature_index] = df.iloc[i, feature_index]

        # random row index to randomly sampled values for each features
        rand_row_index = np.random.choice(
            len(df), len(aug_feature_index), replace=True)

        # for each feature being randomly sampled, extract value from random row in df
        for n, j in enumerate(aug_feature_index):
            aug_df.iloc[i, j] = df.iloc[rand_row_index[n], j]

    return aug_df


if __name__ == '__main__':
    # augment health database
    df = pd.read_csv('data/heart.csv')
    for _ in range(5):
        aug_df = augment_data(df)
        df = pd.concat((df, aug_df))

    df = df.drop('target', axis=1)
    df.to_csv('data/health_database.csv', index=False)

    # generate fake identified cross reference database
    names = [fake.name_male() if x == 1 else fake.name_female()
             for x in df['sex']]  # simulate names
    locations = [fake.address() for _ in range(len(df))]  # simulate location
    ssns = [fake.ssn() for _ in range(len(df))] # ssn

    public_data = pd.DataFrame(list(zip(
        df['sex'].to_list(),
        df['age'].to_list(),
        names,
        locations,
        ssns
    )), columns=['sex', 'age', 'name', 'loc', 'ssn']).sample(frac=1)
    public_data.to_csv('data/crossref_database.csv', index=False)
