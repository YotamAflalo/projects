
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, make_scorer, f1_score, recall_score,precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from numpy import mean

def cancer_encode_without_ms(df,y=None):
    cat_var = ['Race','6th Stage']
    one_hot = OneHotEncoder(sparse=False)  # , drop = 'first')
    encoder_var_array = one_hot.fit_transform(df[cat_var])
    encoder_name = one_hot.get_feature_names_out(cat_var)
    encoder_vars_df = pd.DataFrame(encoder_var_array, columns=encoder_name)
    df = pd.concat([df, encoder_vars_df], axis=1)
    data_encoder = preprocessing.OrdinalEncoder(
        categories=[['T1', 'T2', 'T3', 'T4'], ['N1', 'N2', 'N3'], ['IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC'],
                    ['1', '2', '3', ' anaplastic; Grade IV'], ['Regional', 'Distant'],
                    ['Negative', 'Positive'], ['Negative', 'Positive'], ['Alive', 'Dead']])
    df[['T Stage ', 'N Stage', '6th Stage', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status',
        'Status']] = data_encoder.fit_transform(df[['T Stage ', 'N Stage', '6th Stage', 'Grade', 'A Stage',
                                                    'Estrogen Status', 'Progesterone Status', 'Status']].values.reshape(
        -8, 8))
    df.Grade = df.Grade + 1

    return df

def cancer_encode(df,y=None):

    cat_var = ['Race', 'Marital Status','6th Stage']
    one_hot = OneHotEncoder(sparse=False)  # , drop = 'first')
    encoder_var_array = one_hot.fit_transform(df[cat_var])
    encoder_name = one_hot.get_feature_names_out(cat_var)
    encoder_vars_df = pd.DataFrame(encoder_var_array, columns=encoder_name)
    df = pd.concat([df, encoder_vars_df], axis=1)
    data_encoder = preprocessing.OrdinalEncoder(
        categories=[['T1', 'T2', 'T3', 'T4'], ['N1', 'N2', 'N3'], ['IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC'],
                    ['1', '2', '3', ' anaplastic; Grade IV'], ['Regional', 'Distant'],
                    ['Negative', 'Positive'], ['Negative', 'Positive'], ['Alive', 'Dead']])
    df[['T Stage ', 'N Stage', '6th Stage', 'Grade','A Stage', 'Estrogen Status', 'Progesterone Status',
        'Status']] = data_encoder.fit_transform(df[['T Stage ', 'N Stage', '6th Stage', 'Grade','A Stage',
                                                    'Estrogen Status', 'Progesterone Status', 'Status']].values.reshape(
        -8, 8))

    df.Grade = df.Grade + 1
    return df


def cancer_features_select(df):
    df['Regional_Node_pos_%'] = 100 * df['Reginol Node Positive'] / df['Regional Node Examined']
    df['Estrogen&Progesterone positive'] = df['Estrogen Status'] * df["Progesterone Status"]
    df['Estrogen&Progesterone Negative'] = (1-df['Estrogen Status']) * (1-df["Progesterone Status"])
    df.drop(['Race', 'Marital Status', 'Survival Months', 'Status','differentiate'], axis=1, inplace=True)
    return df



cancer_encoder = FunctionTransformer(cancer_encode, validate=False)
cancer_features_selector = FunctionTransformer(cancer_features_select, validate=False)

def report(clf, X, y):
    acc = accuracy_score(y_true=y,
                         y_pred=clf.predict(X))
    cm = pd.DataFrame(confusion_matrix(y_true=y,
                                       y_pred=clf.predict(X)),
                      index=clf.classes_,
                      columns=clf.classes_)
    rep = classification_report(y_true=y,
                                y_pred=clf.predict(X))
    return 'accuracy: {:.3f}\n\n{}\n\n{}'.format(acc, cm, rep)
def cross_validation_report(df,binary_target,cv_scores):
    print('Mean f1:  %.3f' % mean(cv_scores['test_f1']))
    print('Mean recall: %.3f' % mean(cv_scores['test_recall']))
    print('Mean precision: %.3f' % mean(cv_scores['test_precision']))
    rec = mean(cv_scores['test_recall'])
    pre = mean(cv_scores['test_precision'])
    total_p = df.groupby([binary_target])[binary_target].count()[1]
    total_n = df.groupby([binary_target])[binary_target].count()[0]
    TP = rec*total_p
    FN = (1-rec)*total_p
    FP = ((1-pre)/pre)*TP
    TN = total_n - FP
    print("cross_validation confusion matrix")
    print("     0       1")
    print("0 ",round(TN), "  ", round(FP))
    print("1  ",round(FN), "     ", round(TP))


def f_recall(y_true, y_pred):
    precision = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if recall >0.80:
        answer = 0.8+ f1_score(y_true, y_pred)
        return answer
    else:
        return recall
def recall8_f1_scorer(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if recall >0.80:
        answer = recall+ f1
        return answer
    else:
        return recall

def recall8_precision_scorer(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if recall >0.80:
        answer = recall+ precision
        return answer
    else:
        return recall

if __name__ == '__main__':
    pass

