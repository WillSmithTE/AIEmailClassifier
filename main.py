import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tree1 import predictionGenerator
from util import LABEL_FIELD, DATE_FIELD, ORG_FIELD, MAIL_TYPE_FIELD, TLD_FIELD, DESIGNATION_FIELD, QUANTITATIVE_FIELD_NAMES, ALL_FIELDS, powerset
import statistics

TEST_SIZE = 0.4
NUM_TESTS = 10

tests = dict()

# for fields in powerset(QUANTITATIVE_FIELD_NAMES):

train_df = pd.read_csv('train.csv', index_col=0)
final_df = pd.read_csv('test.csv', index_col=0)

accuracyScore = 0
for _ in range(NUM_TESTS):
    train_x, test_x, train_y, test_y = train_test_split(train_df, train_df[[LABEL_FIELD]], test_size=TEST_SIZE)
    predicted = predictionGenerator(train_x, train_y, test_x)
    # print(metrics.classification_report(test_y, predicted))
    accuracyScore += metrics.accuracy_score(test_y, predicted)
accuracyScore = accuracyScore / NUM_TESTS
tests[', '.join(train_df.columns)] = accuracyScore
print('Average accuracy score over', NUM_TESTS, 'attempts = ', accuracyScore)
train_x, test_x, train_y, test_y = train_test_split(train_df, train_df[[LABEL_FIELD]], test_size=1)
finalPredictions = predictionGenerator(train_x, train_y, final_df)
finalPredictions = pd.DataFrame(finalPredictions, columns=['label'])
finalPredictions.to_csv("knn_sample_submission.csv", index=True, index_label='Id')

print('-----------COMPLETE-----------')

maxCount = 0
maxVariables = ''
mean = statistics.mean(tests.values())
for test in list(tests.items()):
    if test[1] > maxCount:
        maxCount = test[1]
        maxVariables = test[0]

print()
print('mean accuracy:', mean)
print('max accuracy:', maxCount)
presentVariables = maxVariables.split(', ')
for var in ALL_FIELDS:
    tickOrCross = u'\u2713' if var in presentVariables else u'\u2717'
    print('    - ', var, ' ', tickOrCross)
