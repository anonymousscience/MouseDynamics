# Load libraries
import pandas as pd
import settings as st
import csv
import math
import numpy
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d



NUM_NEGATIVE_SAMPLES_PER_CLASS = 200
NUM_POSITIVE_SAMPLES = 1800

# number of users in the Balabit data set
NUM_CLASSES = 10
# parameter for Random Forest
NUM_TREES = 500


def computeAUC( scorefilename ):
    data = pd.read_csv(scorefilename, names=['label','score'])
    # data = pd.read_csv(scorefilename)
    labels = data['label']
    scores = data['score']
    labels = [int(e)   for e in labels]
    scores = [float(e) for e in scores]
    auc_value =   metrics.roc_auc_score(numpy.array(labels), numpy.array(scores) )
    # plot ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    # EER = thresholds(numpy.argmin(abs(tpr - fpr)))
    print("EER: "+str(eer))
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if st.SESSION_CUT == 1:
        title = 'Modeling unit: action sequence. ROC curve.'
    else:
        title = 'Modeling unit: action. ROC curve.'
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    # end plot
    return auc_value


def select_negative_indexes_for_class(array, classid):
    fr = {'7': 0, '9': 0, '12': 0,'15':0, '16':0,'20':0,'21':0,'23':0,'29':0,'35':0}
    negative_indexes = []
    counter = 0
    # ends with class,session,n_from,n_to
    numFeatures = array.shape[1]-3
    # print('numFeatures: '+str(numFeatures))
    # line ends with ...,classid,sessionid
    for row in array:
        row_classid = str(int(row[ numFeatures-1  ]))

        if  row_classid != str(classid) and fr[row_classid] < NUM_NEGATIVE_SAMPLES_PER_CLASS:
            negative_indexes.append( counter )
            fr[row_classid] += 1
        counter += 1
    return negative_indexes

def select_positive_indexes_for_class(array, classid):
    positive_indexes = []
    counter = 0
    positive_counter = 0
    # ends with class,session,n_from,n_to
    numFeatures = array.shape[1]-3
    for row in array:
        row_classid = str(int(row[ numFeatures-1 ]))
        if  positive_counter >= NUM_POSITIVE_SAMPLES:
            return positive_indexes
        if row_classid == str(classid):
            positive_indexes.append( counter)
            positive_counter += 1
        counter += 1
    return positive_indexes

def evaluate_training( filename ):
    dataset = pd.read_csv( filename )
    print("Filename: "+filename )
    numSamples  = int(dataset.shape[0])
    # ends with class,session,n_from,n_to
    numFeatures = int(dataset.shape[1])-3
    print("numSamples: "+str(numSamples)+"\tnumFeatures: "+str(numFeatures))
    classes = dataset.groupby('class')
    # positive_scores = open("positive.txt", "w")
    # negative_scores = open("negative.txt", "w")
    scores = open("output/scores.csv", "w")
    print("User based predictions on the validation set using Random Forest classifier")
    for c in classes:
        classid = c[0]
        numSamples = c[1].shape[0]
        # print("Classid: "+ str(classid)+" NumSamples: "+str(numSamples) )
        # positive_array = c[1].values
        array = dataset.values
        negative_indexes = select_negative_indexes_for_class(array, classid)
        positive_indexes = select_positive_indexes_for_class(array, classid)

        negative_array = array[negative_indexes,:]
        for row in negative_array:
            row[numFeatures-1]=0
        positive_array = array[positive_indexes,:]

        # print("Positive samples:"+str(len(positive_array)))
        # print("Negative samples:" + str(len(negative_array)))

        user_array = numpy.concatenate([positive_array, negative_array])
        X = user_array[:,0:numFeatures -1]
        Y = user_array[:, numFeatures-1]

        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

        # Make predictions on validation dataset
        rf =  RandomForestClassifier(n_estimators=NUM_TREES)
        rf.fit(X_train, Y_train)
        predictions = rf.predict(X_validation)
        userAccuracy = accuracy_score(Y_validation, predictions)
        print("User\t" + str(classid)+"\t" +str(userAccuracy))

        counter = 0
        for x in X_validation:
            res =  rf.predict_proba( x )
            classid = int( Y_validation[counter] )
            if classid == 0:
                scores.write("0,"+str(res[0, 1]) + "\n")
            else:
                scores.write("1," + str(res[0, 1]) + "\n")
            counter += 1
    scores.close()
    print( "Training AUC on validation set: "+str(computeAUC("output/scores.csv")) )
    return

def create_binary_classifiers( feature_filename ):
    dict = {}
    dataset = pd.read_csv(feature_filename)
    # ends with class,session,n_from,n_to
    numFeatures = int(dataset.shape[1]) - 3
    classes = dataset.groupby('class')
    # create binary classifier for each class
    print("Creating binary classifiers")
    for c in classes:
        classid = c[0]
        numSamples = c[1].shape[0]
        # print("Classid: " + str(classid) + " NumSamples: " + str(numSamples))
        print("Classid: " + str(classid) )

        array = dataset.values
        negative_indexes = select_negative_indexes_for_class(array, classid)
        positive_indexes = select_positive_indexes_for_class(array, classid)

        negative_array = array[negative_indexes, :]
        for row in negative_array:
            row[numFeatures-1] = 0
        positive_array = array[positive_indexes, :]
        user_array = numpy.concatenate([positive_array, negative_array])
        print(str(classid)+", "+str(len(positive_array))+","+str(len(negative_array)))

        X = user_array[:, 0:numFeatures - 1]
        Y = user_array[:, numFeatures - 1]

        #
        validation_size = 0.0
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=seed)
        rf = RandomForestClassifier(n_estimators=NUM_TREES)
        rf.fit(X_train, Y_train)

        dict [ classid ] = rf
    return dict

# Computes one score for all the actions belonging to a test session
def evaluate_test_session( training_feat_file, test_feat_file):
    #  create 2-class classifiers
    classifiers = create_binary_classifiers(training_feat_file)
    for c in classifiers:
        print(c)
    # read test feature file

    dataset = pd.read_csv(test_feat_file)
    # number of features used for classification: session and islegal are not used for classification
    # ..., class, session, n_from, n_to, islegal
    numFeatures = int(dataset.shape[1]) - 4
    numSamples = int(dataset.shape[0])

    print("Computing scores for test sessions. Be patient ...")
    userscores = open("output/usertestscores.csv", "w")
    scores = open("output/testscores.csv", "w")
    print("Num samples to evaluate: "+ str(numSamples))

    # Group by
    sessions = dataset.groupby('session')
    sessionCounter = 0
    for name, group in sessions:
        sessionCounter += 1
        if sessionCounter % 100 == 0:
            print("Sessioncounter: " + str(sessionCounter))
        array = numpy.array(group)
        n = array.shape[0]
        probs = []
        classid = array[0, numFeatures - 1]
        label = array[0, numFeatures + 3]
        for i in range(0,n):
            resprob = classifiers[classid].predict_proba(array[i, 0:numFeatures-1])
            probs.append( resprob[0, 1] )
        prob = numpy.mean(probs)
        scores.write( str(label)+"," + str(prob) + "\n")
        userscores.write( str(classid) + ',' + str(label) + ',' + str(resprob[0, 1]) + "\n")

        # print( '\t'+str(sessionCounter) +', '+str(prob))
    scores.close()
    userscores.close()
    print("Test AUC:" + str(computeAUC("output/testscores.csv")))
    return

# Computes score for the sequence of actions belonging to a test session
def evaluate_test_actions( training_feat_file, test_feat_file, num_actions):
    #  create 2-class classifiers
    classifiers = create_binary_classifiers(training_feat_file)
    for c in classifiers:
        print(c)
    # read test feature file
    dataset = pd.read_csv(test_feat_file)
    # number of features used for classification: session and islegal are not used for classification
    # ..., class, session, n_from, n_to, islegal
    numFeatures = int(dataset.shape[1]) - 4
    numSamples = int(dataset.shape[0])

    print("Computing scores for test sessions ...")
    userscores = open("output/usertestscores.csv", "w")
    scores = open("output/testscores.csv", "w")
    print("Num samples to evaluate: "+ str(numSamples))

    # Group by
    sessions = dataset.groupby('session')
    sessionCounter = 0
    for name, group in sessions:
        sessionCounter += 1
        if sessionCounter % 100 == 0:
            print("Sessioncounter: "+str(sessionCounter))
        array = numpy.array(group)
        n = array.shape[0]
        classid = array[0, numFeatures - 1]
        label = array[0, numFeatures + 3]
        probs = []
        for i in range(0,n):
            resprob = classifiers[classid].predict_proba(array[i, 0:numFeatures - 1])
            if len(probs) == num_actions:
                prob = numpy.mean(probs)
                userscores.write(str(classid) + ','+str(label)+',' + str(resprob[0, 1]) + "\n")
                scores.write( str(label)+"," + str(prob) + "\n")
                # print('\t' + str(sessionCounter) + ', ' + str(prob))
                probs = []
            probs.append(resprob[0, 1])
    scores.close()
    userscores.close()
    print("Test AUC:" + str(computeAUC("output/testscores.csv")))
    return



# computes scores for each feature vector
def evaluate_test_actions2( training_feat_file, test_feat_file ):
    #  create 2-class classifiers
    classifiers  = create_binary_classifiers( training_feat_file )
    for c in classifiers:
         print( c )
    # read test feature file

    dataset = pd.read_csv(test_feat_file)
    # number of features used for classification: session and islegal are not used for classification
    # ..., class, session, n_from, n_to, islegal
    numFeatures = int(dataset.shape[1]) - 4
    numSamples = int(dataset.shape[0])

    array = dataset.values
    X = array[:, 0:numFeatures - 1]
    #  class
    Y = array[:, numFeatures - 1]
    session = array[:, numFeatures ]
    labels  = array[:, numFeatures+3 ]

    print("Computing scores for test sessions ...")
    scores = open("output/testscores.csv", "w")
    userscores = open("output/usertestscores.csv", "w")
    for i in range(0,numSamples):
        classid = Y[i]
        resprob = classifiers[classid].predict_proba(X[i, :])
        if labels[i] == 0:
            scores.write("0," + str( resprob[0,1]) + "\n")
            userscores.write(str(classid)+",0," + str(resprob[0, 1]) + "\n")
        else:
            scores.write("1," + str( resprob[0,1]) + "\n")
            userscores.write(str(classid)+",1," + str(resprob[0, 1]) + "\n")
    scores.close()
    userscores.close()
    print("Test AUC:"+ str(computeAUC("output/testscores.csv")))
    return

# Computes score for the sequence of actions belonging to a test session
# only sessions containing at least numactions
def evaluate_test_session_having_at_least( training_feat_file, test_feat_file, numactions):
    #  create 2-class classifiers
    classifiers = create_binary_classifiers(training_feat_file)
    for c in classifiers:
        print(c)
    # read test feature file

    dataset = pd.read_csv(test_feat_file)
    # number of features used for classification: session and islegal are not used for classification
    # ..., class, session, n_from, n_to, islegal
    numFeatures = int(dataset.shape[1]) - 4
    numSamples = int(dataset.shape[0])

    print("Computing scores for test sessions. Be patient ...")
    userscores = open("output/usertestscores.csv", "w")
    scores = open("output/testscores.csv", "w")
    print("Num samples to evaluate: "+ str(numSamples))

    # Group by
    sessions = dataset.groupby('session')
    sessionCounter = 0
    unused = 0
    for name, group in sessions:
        sessionCounter += 1
        array = numpy.array(group)
        n = array.shape[0]
        # print("numactions "+str(n))
        if n< numactions:
            unused += 1
            continue
        probs = []
        classid = array[0, numFeatures - 1]
        label = array[0, numFeatures + 3]
        for i in range(0,n):
            resprob = classifiers[classid].predict_proba(array[i, 0:numFeatures-1])
            probs.append( resprob[0, 1] )
        prob = numpy.mean(probs)
        scores.write( str(label)+"," + str(prob) + "\n")
        userscores.write( str(classid) + ',' + str(label) + ',' + str(resprob[0, 1]) + "\n")

        # print( '\t'+str(sessionCounter) +', '+str(prob))
    scores.close()
    userscores.close()
    print("Test AUC:" + str(computeAUC("output/testscores.csv")))
    used = 816 - unused
    print("Num sessions: "+
          str(used))
    return



def read_labels():
    with open("output/publiclabels.txt","r") as ins:
        array = []
        for line in ins:
            array.append(int(line))
    return array





