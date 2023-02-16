# imports
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


def randSampler(data):
    Y = data['Label']
    x = data.drop('Label', 1)
    underSampler = RandomUnderSampler(random_state=0)
    underSamplerXY = underSampler.fit_resample(x, Y)

    overSampler = RandomOverSampler(random_state=0)
    overSamplerXY = overSampler.fit_resample(x, Y)

    overSampler = SMOTE(random_state=0)
    overSampler_SMOTE_XY = overSampler.fit_resample(x, Y)

    return underSamplerXY, overSamplerXY, overSampler_SMOTE_XY


# fix nan data(get dataframe and return new dataframe with no nan data)
def formatData(data):
    newDF = pd.DataFrame()  # creates a new dataframe that's empty
    for col_name in data.columns:
        newDF[col_name] = data[col_name].fillna(data[col_name].mode()[0])
        # check nulls
        # print("column:", col_name, ".Missing:", sum(data[col_name].isnull()))
    return newDF


def naive(xTrain, xTest, yTrain, yTest):
    naiveBayes = GaussianNB()
    naiveBayes.fit(xTrain, yTrain)
    y_pred_naiveBayes = naiveBayes.predict(xTest)
    accuracy_naiveBayes = accuracy_score(yTest, y_pred_naiveBayes)
    return print('accuracy naive bayes: ', accuracy_naiveBayes * 100)


def onn(xTrain, xTest, yTrain, yTest):
    Onn = KNeighborsClassifier(n_neighbors=1)
    Onn.fit(xTrain, yTrain)
    y_pred_Onn = Onn.predict(xTest)
    accuracy_Onn = accuracy_score(yTest, y_pred_Onn)
    return print('accuracy ONN: ', accuracy_Onn * 100)


def linSvc(xTrain, xTest, yTrain, yTest):
    clf = LinearSVC()
    clf.fit(xTrain, yTrain)
    y_pred_svc = clf.predict(xTest)
    accuracy_svc = accuracy_score(yTest, y_pred_svc)
    return print('accuracy lin svc: ', accuracy_svc * 100)


def svmRbf(xTrain, xTest, yTrain, yTest):
    clf = svm.SVC(kernel='rbf')
    clf.fit(xTrain, yTrain)
    y_pred_svm = clf.predict(xTest)
    accuracy_svm = accuracy_score(yTest, y_pred_svm)
    return print('accuracy smv with kernel rbf: ', accuracy_svm * 100)


def main():
    ConPath = 'Covid-19.csv'
    # read data
    df = pd.read_csv(ConPath)
    noNan_data = formatData(df)
    newXY_underSample, newXY_overSample, newXY_SMOTE_overSample = randSampler(noNan_data)
    xTrainUnder, xTestUnder, yTrainUnder, yTestUnder = train_test_split(newXY_underSample[0], newXY_underSample[1],
                                                                        test_size=0.3,
                                                                        shuffle=True)
    print("++++++++++++++++++++++ underSampler ++++++++++++++++++++++")

    naive(xTrainUnder, xTestUnder, yTrainUnder, yTestUnder)
    onn(xTrainUnder, xTestUnder, yTrainUnder, yTestUnder)
    linSvc(xTrainUnder, xTestUnder, yTrainUnder, yTestUnder)
    svmRbf(xTrainUnder, xTestUnder, yTrainUnder, yTestUnder)

    xTrainOver, xTestOver, yTrainOver, yTestOver = train_test_split(newXY_overSample[0], newXY_overSample[1],
                                                                    test_size=0.3,
                                                                    shuffle=True)
    print("++++++++++++++++++++++ overSampler ++++++++++++++++++++++")

    naive(xTrainOver, xTestOver, yTrainOver, yTestOver)
    onn(xTrainOver, xTestOver, yTrainOver, yTestOver)
    linSvc(xTrainOver, xTestOver, yTrainOver, yTestOver)
    svmRbf(xTrainOver, xTestOver, yTrainOver, yTestOver)

    xTrainOver_SMOTTE, xTestOver_SMOTTE, yTrainOver_SMOTTE, yTestOver_SMOTTE = train_test_split(
        newXY_SMOTE_overSample[0], newXY_SMOTE_overSample[1], test_size=0.3,
        shuffle=True)
    print("++++++++++++++++++++++ overSampler SMOTE ++++++++++++++++++++++")

    naive(xTrainOver_SMOTTE, xTestOver_SMOTTE, yTrainOver_SMOTTE, yTestOver_SMOTTE)
    onn(xTrainOver_SMOTTE, xTestOver_SMOTTE, yTrainOver_SMOTTE, yTestOver_SMOTTE)
    linSvc(xTrainOver_SMOTTE, xTestOver_SMOTTE, yTrainOver_SMOTTE, yTestOver_SMOTTE)
    svmRbf(xTrainOver_SMOTTE, xTestOver_SMOTTE, yTrainOver_SMOTTE, yTestOver_SMOTTE)


if __name__ == '__main__':
    main()
