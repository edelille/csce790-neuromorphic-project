import json
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

TF_IDF_ENC_PATH = 'encodings/tf-idf_FLAT_encoding.txt'
BOOL_ENC_PATH = 'encodings/bool_FLAT_encoding.txt'
DATA_PATH = 'data/tf-idf.xlsx'

X_tf_idf = []
X_bool = []
Y = []
np.random.seed(100)

class_num = {
    'no': 0,
    'yes': 1
}

def num(arg):
    if arg is None:
        return 0
    else:
        return arg

def get_data(row, cw_list):

    global X_tf_idf
    global X_bool
    global Y

    word_counts = json.loads(row['tf_idf'])
    tf_idf = []
    bool = []
    for cw in cw_list[0]:
        tf_idf.append(num(word_counts.get(cw)))
    for cw in cw_list[1]:
        bool.append(0 if num(word_counts.get(cw)) == 0 else 1)
    X_tf_idf.append(tf_idf)
    X_bool.append(bool)
    Y.append(class_num[row['class']])
    return row

def main():

    print('Loading data...')
    df = pd.read_excel(DATA_PATH, engine='openpyxl')
    cw_tf_idf = []
    cw_bool = []
    r = open(TF_IDF_ENC_PATH)
    for line in r.readlines():
        line = line.strip()
        if line != '':
            cw_tf_idf.append(line)
    r.close()
    r = open(BOOL_ENC_PATH)
    for line in r.readlines():
        line = line.strip()
        if line != '':
            cw_bool.append(line)

    print('Getting tf-idf and bool vectors...')
    df.apply(lambda row: get_data(row, [cw_tf_idf, cw_bool]), axis=1)

    print('Splitting tf-idf and bool training data and validation data...')
    X_tf_idf_train, X_tf_idf_test, Y_tf_idf_train, Y_tf_idf_test = train_test_split(X_tf_idf, Y, test_size=0.2)
    X_bool_train, X_bool_test, Y_bool_train, Y_bool_test = train_test_split(X_bool, Y, test_size=0.2)

    print('Fitting tf-idf SVM with vectors...')
    clf_tf_idf = svm.SVC()
    clf_tf_idf.fit(X_tf_idf_train, Y_tf_idf_train)

    print('Fitting bool SVM with vectors...')
    clf_bool = svm.SVC()
    clf_bool.fit(X_bool_train, Y_bool_train)

    print('Evaluating the tf-idf model...')
    Y_pred = clf_tf_idf.predict(X_tf_idf_test)
    print('Accuracy:', metrics.accuracy_score(Y_tf_idf_test, Y_pred))
    print('Precision:', metrics.precision_score(Y_tf_idf_test, Y_pred))
    print('Recall:', metrics.recall_score(Y_tf_idf_test, Y_pred))

    print('Evaluating the bool model...')
    Y_pred = clf_bool.predict(X_bool_test)
    print('Accuracy:', metrics.accuracy_score(Y_bool_test, Y_pred))
    print('Precision:', metrics.precision_score(Y_bool_test, Y_pred))
    print('Recall:', metrics.recall_score(Y_bool_test, Y_pred))

if __name__ == '__main__':

    print('Starting svm...')
    main()
    