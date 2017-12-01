# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

import settings as st

def feature_selection_univariate():
    # load data
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    # dataframe = pandas.read_csv(url, names=names)

    filename = st.TRAINING_FEATURE_FILENAME
    dataframe = pandas.read_csv(filename)
    array = dataframe.values
    X  = array[:,0:32]
    Y = array[:,32]
    # feature extraction
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, Y)
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    # summarize selected features
    print(features[0:5,:])
    return

def feature_selection_importance():
    filename = st.TRAINING_FEATURE_FILENAME
    dataframe = pandas.read_csv(filename)
    array = dataframe.values
    X = array[:, 0:32]
    Y = array[:, 32]
    # feature extraction
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    return