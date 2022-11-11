import pandas as pd
import numpy as np
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

fake = Faker()

if __name__ == '__main__':
	df_census = pd.read_csv('data/New_York_NY.csv')
	size = len(df_census)
	names = [fake.name() for _ in range(size)]
	addresses = [fake.address() for _ in range(size)]
	df_census['name'] = names
	df_census['loc'] = addresses
	print(df_census.head())
	le = LabelEncoder()
	y = le.fit_transform(df_census['name'])
	X = df_census[['REGION', 'P0050007', 'P0050008', 'GEOVAR']]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

	model = LogisticRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print(accuracy_score(y_test, y_pred))