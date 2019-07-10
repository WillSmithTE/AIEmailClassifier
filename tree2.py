from sklearn import tree
from util import CCS_FIELD, BCCED_FIELD, IMAGES_FIELD, URLS_FIELD, SALUTATIONS_FIELD, DESIGNATION_FIELD, CHARS_IN_SUBJECT_FIELD, CHARS_IN_BODY_FIELD, MAIL_TYPE_FIELD, DATE_FIELD, ORG_FIELD, TLD_FIELD, LABEL_FIELD
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def predictionGenerator(train_x, train_y, test_x):

    for dataSet in [ train_x, test_x ]:
        dataSet[MAIL_TYPE_FIELD] = dataSet[MAIL_TYPE_FIELD].apply(lambda value: value.lower())
        # del dataSet[DATE_FIELD]
        # del dataSet[ORG_FIELD]
        # del dataSet[MAIL_TYPE_FIELD]
        # del dataSet[TLD_FIELD]
        if LABEL_FIELD in dataSet.columns:
            del dataSet[LABEL_FIELD]
        
        for feature in [ ORG_FIELD, MAIL_TYPE_FIELD, TLD_FIELD]:

                dataSet[feature] = pd.Categorical(dataSet[feature])
                dummies = pd.get_dummies(dataSet[MAIL_TYPE_FIELD], prefix='mail-type')

                dataSet = pd.concat([ dataSet, dummies ], axis=1)
                del dataSet[feature]    

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(train_x, train_y)
    
    prediction = classifier.predict(test_x)

    return prediction