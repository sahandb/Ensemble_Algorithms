# imports
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc


# def randUndSampler(data):
#     Y = data['Label']
#     x = data.drop('Label', 1)
#     underSampler = RandomUnderSampler(random_state=0)
#     fitted = underSampler.fit_resample(x, Y)
#     return fitted

def randUndSampler(data):
    len_class_minority = len(data.loc[data['Label'] == 1])
    data_underSample_majority = data.loc[data['Label'] == -1].sample(n=len_class_minority, random_state=1)
    underSampled = pd.concat([data.loc[data['Label'] == 1], data_underSample_majority])
    Y = underSampled['Label']
    x = underSampled.drop('Label', 1)
    return x, Y


# fix nan data(get dataframe and return new dataframe with no nan data)
def formatData(data):
    newDF = pd.DataFrame()  # creates a new dataframe that's empty
    for col_name in data.columns:
        newDF[col_name] = data[col_name].fillna(data[col_name].mode()[0])
        # check nulls
        # print("column:", col_name, ".Missing:", sum(data[col_name].isnull()))
    return newDF


# # A function that takes classifier as input and performs classification
# def generic_clf(Y_train, X_train, Y_test, X_test, clf):
#     clf.fit(X_train, Y_train)
#     pred_train = clf.predict(X_train)
#     pred_test = clf.predict(X_test)
#     return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test)


# Returns accuracy of given the prediction and the input
def get_acc_rate(pred, Y):
    return sum(pred == Y) / float(len(Y))


def bagging(run, path):
    trees = list()
    pred_test_list = list()
    y_pred_count_list = list()

    df = pd.read_csv(path)
    noNan_data = formatData(df)
    newX, newY = randUndSampler(noNan_data)

    # Split into training and test set
    x_train, x_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3)

    for iter in range(run):
        # Fit a simple decision tree first
        clf = DecisionTreeClassifier(random_state=1)
        clf.fit(x_train, y_train)
        trees.append(clf)
        pred_test = clf.predict(x_test)
        pred_test_list.append(pred_test)
    y_pred_lisT = np.asarray(pred_test_list)
    y_pred_vote = np.asarray(pd.DataFrame(y_pred_lisT).mode(axis=0))  # vote over all run
    # for item in range(len(y_test)):

    return y_pred_vote.reshape(y_pred_vote.shape[1]), y_test.values


def run(times, runn):
    precision_list = list()
    recall_list = list()
    f1_list = list()
    auc_list = list()
    accuracy_list = list()
    g_mean_list = list()
    for tim in range(times):
        a, b = bagging(runn, 'Covid-19.csv')

        # minor class reports
        convert_ypred = 1 * (a == 1)
        convert_y_2to1 = 1 * (b == 1)
        report = prfs(convert_y_2to1, convert_ypred, average='binary')
        fpr, tpr, thresh = roc_curve(convert_y_2to1, convert_ypred)
        report_auc = auc(fpr, tpr)
        report_accuracy = get_acc_rate(a, b) * 100
        gmean = np.sqrt(np.multiply(tpr, fpr)) * 100

        precision_list.append(report[0])
        recall_list.append(report[1])
        f1_list.append(report[2])
        auc_list.append(report_auc)
        accuracy_list.append(report_accuracy)
        g_mean_list.append(gmean[1])
        print('for run: {0} Precision = {1}, Recall = {2}, F1 = {3}, AUC = {4}, Accuracy = {5}, G-mean = {6} \n'.format(
            tim + 1,
            report[0] * 100,
            report[1] * 100,
            report[2] * 100,
            report_auc,
            report_accuracy,
            gmean[1]))

    return precision_list, recall_list, f1_list, auc_list, accuracy_list, g_mean_list


def main():
    sets = [11, 31, 51, 101]
    # sets = [11]
    runners = 10
    for sett in sets:
        print("+++++++++++++++++++++++++++++++++++++++ set: {0} +++++++++++++++++++++++++++++++++++++++".format(sett))
        precision, recall, f1, auc, accuracy, g_mean = run(runners, sett)
        print("============================= mean set: {0} =============================".format(sett))
        print('for set: {0} Precision = {1}, Recall = {2}, F1 = {3}, AUC = {4}, Accuracy = {5}, G-mean = {6} \n'.format(
            sett,
            np.mean(precision),
            np.mean(recall),
            np.mean(f1),
            np.mean(auc),
            np.mean(accuracy),
            np.mean(g_mean)))
        print("============================= std set: {0} =============================".format(sett))
        print('for set: {0} Precision = {1}, Recall = {2}, F1 = {3}, AUC = {4}, Accuracy = {5}, G-mean = {6} \n'.format(
            sett,
            np.std(precision),
            np.std(recall),
            np.std(f1),
            np.std(auc),
            np.std(accuracy),
            np.std(g_mean)))


# Driver
if __name__ == '__main__':
    main()
