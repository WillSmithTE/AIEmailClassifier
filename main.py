import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tree import predictionGenerator
from util import LABEL_FIELD

TEST_SIZE = 0.4
NUM_TESTS = 10

train_df = pd.read_csv('train.csv', index_col=0)
final_df = pd.read_csv('test.csv', index_col=0)

accuracyScore = 0

print(train_df['mail_type'].unique().tolist())
print(final_df['mail_type'].unique().tolist())

for _ in range(NUM_TESTS):
    train_x, test_x, train_y, test_y = train_test_split(train_df, train_df[[LABEL_FIELD]], test_size=TEST_SIZE)
    predicted = predictionGenerator(train_x, train_y, test_x)
    print(metrics.classification_report(test_y, predicted))
    accuracyScore += metrics.accuracy_score(test_y, predicted)

print('Average accuracy score over', NUM_TESTS, 'attempts = ', accuracyScore/NUM_TESTS)

train_x, test_x, train_y, test_y = train_test_split(train_df, train_df[[LABEL_FIELD]], test_size=1)
finalPredictions = predictionGenerator(train_x, train_y, final_df)

finalPredictions = pd.DataFrame(finalPredictions, columns=['label'])
finalPredictions.to_csv("knn_sample_submission.csv", index=True, index_label='Id')

print('-----------COMPLETE-----------')