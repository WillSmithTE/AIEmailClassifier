import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skeleton_code import predictionGenerator

TEST_SIZE = 0.4
LABEL_FIELD = 'label'

train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

train_x, test_x, train_y, test_y = train_test_split(train_df, train_df[[LABEL_FIELD]], test_size=TEST_SIZE)

predicted = predictionGenerator(train_x, train_y, test_x)

print(metrics.classification_report(test_y, predicted))