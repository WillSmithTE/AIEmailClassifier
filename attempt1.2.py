"""
This script can be used as skelton code to read the challenge train and test
csvs, to train a trivial model, and write data to the submission file.
"""
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer


def testResults(data):
	print(data)	
	
## Read csvs

train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

train_x, test_y, test_x, test_y = train_test_split(train_df, train_df[['label']], test_size=0.4)

columnTransformer = ColumnTransformer([("herpy derp", OneHotEncoder(), [ 'org', 'tld', 'mail_type'])], remainder='passthrough')

columnTransformer.fit(train_x)
train_x_featurized = columnTransformer.transform(train_x)
test_x_featurized = columnTransformer.transform(test_x)

## Train a simple KNN classifier using featurized data
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x_featurized, train_y)
pred_y = neigh.predict(test_x_featurized)

## Save results to submission file
pred_df = pd.DataFrame(pred_y, columns=['label'])
pred_df.to_csv("knn_sample_submission", index=True, index_label='Id')

testResults(test_df)


