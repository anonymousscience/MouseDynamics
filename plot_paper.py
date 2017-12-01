import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import plot_paper as bsp
import general_statistics as gs

def plotROCCase1( scorefilename ):
    data_no = pd.read_csv(scorefilename, names=['label','score'])
    labels_no = data_no['label']
    scores_no = data_no['score']
    labels_no = [int(e)   for e in labels_no]
    scores_no = [float(e) for e in scores_no]
    auc_value_no =   metrics.roc_auc_score(np.array(labels_no), np.array(scores_no) )

    fpr_no, tpr_no, thresholds_no = metrics.roc_curve(labels_no, scores_no, pos_label=1)
    eer_no = brentq(lambda x: 1. - x - interp1d(fpr_no, tpr_no)(x), 0., 1.)
    # thresh_no = interp1d(fpr_no, thresholds_no)(eer_no)

    plt.figure()
    lw = 2
    plt.plot(fpr_no,     tpr_no,     color='black', lw=lw, label='AUC = %0.4f' % auc_value_no)
    plt.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Modeling unit: sequence of actions. ROC curve.')
    plt.legend(loc="lower right")
    plt.show()
    return



def plotROCCase2( scorefilename ):
    data_no = pd.read_csv(scorefilename, names=['label','score'])
    labels_no = data_no['label']
    scores_no = data_no['score']
    labels_no = [int(e)   for e in labels_no]
    scores_no = [float(e) for e in scores_no]
    auc_value_no =   metrics.roc_auc_score(np.array(labels_no), np.array(scores_no) )

    fpr_no, tpr_no, thresholds_no = metrics.roc_curve(labels_no, scores_no, pos_label=1)
    eer_no = brentq(lambda x: 1. - x - interp1d(fpr_no, tpr_no)(x), 0., 1.)
    # thresh_no = interp1d(fpr_no, thresholds_no)(eer_no)

    plt.figure()
    lw = 2
    plt.plot(fpr_no,     tpr_no,     color='black', lw=lw, label='AUC = %0.4f' % auc_value_no)
    plt.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Modeling unit: single action. ROC curve.')
    plt.legend(loc="lower right")
    plt.show()
    return




def plotROCs( ):
    # NO
    scorefilename_no = 'output/scores_no.csv'
    data_no = pd.read_csv(scorefilename_no, names=['label','score'])
    labels_no = data_no['label']
    scores_no = data_no['score']
    labels_no = [int(e)   for e in labels_no]
    scores_no = [float(e) for e in scores_no]
    auc_value_no =   metrics.roc_auc_score(np.array(labels_no), np.array(scores_no) )

    fpr_no, tpr_no, thresholds_no = metrics.roc_curve(labels_no, scores_no, pos_label=1)
    eer_no = brentq(lambda x: 1. - x - interp1d(fpr_no, tpr_no)(x), 0., 1.)
    thresh_no = interp1d(fpr_no, thresholds_no)(eer_no)

    # LINEAR
    scorefilename_linear = 'output/scores_linear.csv'
    data_linear = pd.read_csv(scorefilename_linear, names=['label', 'score'])
    labels_linear = data_linear['label']
    scores_linear = data_linear['score']
    labels_linear = [int(e) for e in labels_linear]
    scores_linear = [float(e) for e in scores_linear]
    auc_value_linear = metrics.roc_auc_score(np.array(labels_linear), np.array(scores_linear))

    fpr_linear, tpr_linear, thresholds_linear = metrics.roc_curve(labels_linear, scores_linear, pos_label=1)
    eer_linear = brentq(lambda x: 1. - x - interp1d(fpr_linear, tpr_linear)(x), 0., 1.)
    thresh_linear = interp1d(fpr_linear, thresholds_linear)(eer_linear)


    # SPLINE
    scorefilename_spline = 'output/scores_spline.csv'
    data_spline = pd.read_csv(scorefilename_spline, names=['label', 'score'])
    labels_spline = data_spline['label']
    scores_spline = data_spline['score']
    labels_spline = [int(e) for e in labels_spline]
    scores_spline = [float(e) for e in scores_spline]
    auc_value_spline = metrics.roc_auc_score(np.array(labels_spline), np.array(scores_spline))

    fpr_spline, tpr_spline, thresholds_spline = metrics.roc_curve(labels_spline, scores_spline, pos_label=1)
    eer_spline = brentq(lambda x: 1. - x - interp1d(fpr_spline, tpr_spline)(x), 0., 1.)
    thresh_spline = interp1d(fpr_spline, thresholds_spline)(eer_spline)


    plt.figure()
    lw = 2
    plt.plot(fpr_no,     tpr_no,     color='r', lw=lw, label='NO interp. (AUC = %0.4f)' % auc_value_no)
    plt.plot(fpr_linear, tpr_linear, color='g', lw=lw, label='Linear interp. (AUC = %0.4f)' % auc_value_linear)
    plt.plot(fpr_spline, tpr_spline, color='b', lw=lw, label='Spline interp. (AUC = %0.4f)' % auc_value_spline)

    plt.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves ')
    plt.legend(loc="lower right")
    plt.show()
    # end plot
    return





def plotHistogramsTogether(x, y, titleStr, xlabelStr, ylabelStr):
    min_value = 1
    max_value = 100

    bins = np.linspace(min_value, max_value, 100)
    plt.hist(x, bins, alpha =0.5,color = 'red')
    plt.hist(y, bins, alpha=0.5, color ='green')
    plt.title(titleStr)
    plt.xlabel(xlabelStr)
    plt.ylabel(ylabelStr)
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch =  mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch , green_patch])
    plt.show()
    return


def plotScoresCase1():
    scorefilename = 'output/testscores_case1.csv'
    title = 'Modeling unit: sequence of actions. Score distributions for the test data'
    plotScores(scorefilename, title)
    return

def plotScoresCase2():
    scorefilename = 'output/testscores_case2.csv'
    title = 'Modeling unit: single action. Score distributions for the test data'
    plotScores(scorefilename, title)
    return

def plotScores( scorefilename, title ):
    data = pd.read_csv(scorefilename, names=['label', 'score'])
    df = pd.DataFrame(data)
    positive=df.query('label==1')
    negative=df.query('label==0')
    positive_scores = positive['score']
    negative_scores = negative['score']

    min_value = 1
    max_value = 100

    # bins = np.linspace(min_value, max_value, 100)
    bins = 100

    plt.hist(negative_scores, bins, alpha=0.5, color='red')
    plt.hist(positive_scores, bins, alpha=0.5, color='green')
    plt.title(title)
    plt.xlabel('Score value')
    plt.ylabel('#Occurences')
    red_patch = mpatches.Patch(color='red', label='Negative')
    green_patch = mpatches.Patch(color='green', label='Positive')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    return


def plotHistogram(x, titleStr, xlabelStr, ylabelStr):
    min_value = 0
    max_value = 180
    bins = np.linspace(min_value, max_value, 50)
    plt.hist(x, bins, alpha =0.5,color = 'lightgreen')
    plt.title(titleStr)
    plt.xlabel(xlabelStr)
    plt.ylabel(ylabelStr)
    plt.show()
    return

# input: output/test_sessions_numactions.csv
def plotNumActionsHistogram():
    data = pd.read_csv('output/test_sessions_numactions.csv', names=['session', 'numactions'])
    numactions = data['numactions']
    plotHistogram(numactions,'Number of actions in test sessions','Number of actions', 'Occurrence')
    return

def plotUserHistograms():
    input_training_file = "output/balabit_features_training.csv"
    dataset = pd.read_csv(input_training_file)
    header = dataset.columns.values
    classes = dataset.groupby('class')
    fig = plt.figure()
    # st = fig.suptitle('Traveled distance')
    counter = 0
    for c in classes:
        counter += 1
        classid = c[0]
        numSamples = c[1].shape[0]
        columnData = c[1]['traveled_distance_pixel']
        # print(columnData)
        print("Classid: " + str(classid) + " NumSamples: " + str(numSamples))
        sub1 = fig.add_subplot(2, 5, counter)
        min_value = 0
        max_value = 2000
        bins = np.linspace(min_value, max_value, 21)
        sub1.hist(columnData, bins, alpha=0.5, color='red')
        # plt.subplots_adjust(hspace=.1)
        plt.subplots_adjust(left=1, bottom=1, right=10, top=10, wspace=0, hspace=1)
        axes = plt.gca()
        # axes.set_xlim([xmin, xmax])
        # axes.set_ylim([0, 1])
        # axes.set_xlim([0, 2])
        sub1.set_title("User id:" + str(classid))
        sub1.set_xlabel('Path (pixels)')
        sub1.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    return


def plotTrainingTestFeaturesStatistics1():
    input_training_file = "output/balabit_features_training.csv"
    input_test_file = "output/balabit_features_test.csv"
    training = pd.read_csv(input_training_file)
    test = pd.read_csv(input_test_file)

    fig = plt.figure()
    # traveled_distance
    sub4 = fig.add_subplot(221)
    direction_training = training['traveled_distance_pixel']
    direction_test = test['traveled_distance_pixel']
    min_value = 1
    max_value = 2000
    bins = np.linspace(min_value, max_value, 11)
    sub4.hist(direction_training, bins, alpha=0.5, color='red')
    sub4.hist(direction_test, bins, alpha=0.5, color='green')
    sub4.set_title("Traveled distance")
    sub4.set_xlabel("Traveled distance (pixels)")
    sub4.set_ylabel("Occurrence")
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, green_patch])

    # straightness
    sub6 = fig.add_subplot(222)
    trainc = training['straightness']
    testc = test['straightness']
    min_value = 0
    max_value = 1
    bins = np.linspace(min_value, max_value, 11)
    sub6.hist(trainc, bins, alpha=0.5, color='red')
    sub6.hist(testc, bins, alpha=0.5, color='green')
    sub6.set_title("Straightness")
    sub6.set_xlabel("Straightness (ratio)")
    sub6.set_ylabel("Occurrence")
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, green_patch])

    # mean_curv
    sub8 = fig.add_subplot(223)
    trainc = training['mean_curv']
    testc = test['mean_curv']
    min_value = -0.5
    max_value = 0.5
    bins = np.linspace(min_value, max_value, 11)
    sub8.hist(trainc, bins, alpha=0.5, color='red')
    sub8.hist(testc, bins, alpha=0.5, color='green')
    sub8.set_title("Curvature (mean)")
    sub8.set_xlabel("Curvature (radian/pixels)")
    sub8.set_ylabel("Frequency")
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, green_patch])

    # max_v
    sub9 = fig.add_subplot(224)
    trainc = training['max_v']
    testc = test['max_v']
    min_value = 0
    max_value = 5000
    bins = np.linspace(min_value, max_value, 11)
    sub9.hist(trainc, bins, alpha=0.5, color='red')
    sub9.hist(testc, bins, alpha=0.5, color='green')
    sub9.set_title("Velocity (max)")
    sub9.set_xlabel("Velocity (pixels/sec)")
    sub9.set_ylabel("Frequency")
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, green_patch])

    plt.tight_layout()
    plt.show()
    return



def plotTrainingTestFeaturesStatistics2():
    input_training_file = "output/balabit_features_training.csv"
    input_test_file = "output/balabit_features_test.csv"
    training = pd.read_csv(input_training_file)
    test = pd.read_csv(input_test_file)

    fig = plt.figure()
    # mean_omega
    sub1 = fig.add_subplot(221)
    numpoints_training = training['mean_omega']
    numpoints_test = test['mean_omega']
    min_value = -25
    max_value = 25
    bins = np.linspace(min_value, max_value, 11)
    sub1.hist(numpoints_training, bins, alpha=0.5, color='red')
    sub1.hist(numpoints_test, bins, alpha=0.5, color='green')
    sub1.set_title("Angular velocity (mean)")
    sub1.set_xlabel("Angular velocity (rad/s)")
    sub1.set_ylabel("Frequency")
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, green_patch])

    # dist_end_to_end_line
    sub2 = fig.add_subplot(222)
    numpoints_training = training['dist_end_to_end_line']
    numpoints_test = test['dist_end_to_end_line']
    min_value = 1
    max_value = 1500
    bins = np.linspace(min_value, max_value, 11)
    sub2.hist(numpoints_training, bins, alpha=0.5, color='red')
    sub2.hist(numpoints_test, bins, alpha=0.5, color='green')
    sub2.set_title("Distance from the end to end line")
    sub2.set_xlabel("Distance (pixels)")
    sub2.set_ylabel("Frequency")
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, green_patch])

    # num_critical_points
    sub3 = fig.add_subplot(223)
    numpoints_training = training['num_critical_points']
    numpoints_test = test['num_critical_points']
    min_value = 1
    max_value = 20
    bins = np.linspace(min_value, max_value, 11)
    sub3.hist(numpoints_training, bins, alpha=0.5, color='red')
    sub3.hist(numpoints_test, bins, alpha=0.5, color='green')
    sub3.set_title("Critical points")
    sub3.set_xlabel("Number of critical points")
    sub3.set_ylabel("Frequency")
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, green_patch])

    # largest_deviation
    sub3 = fig.add_subplot(224)
    numpoints_training = training['largest_deviation']
    numpoints_test = test['largest_deviation']
    min_value = 1
    max_value = 1000
    bins = np.linspace(min_value, max_value, 11)
    sub3.hist(numpoints_training, bins, alpha=0.5, color='red')
    sub3.hist(numpoints_test, bins, alpha=0.5, color='green')
    sub3.set_title("Largest deviation")
    sub3.set_xlabel("Largest deviation (pixels)")
    sub3.set_ylabel("Frequency")
    red_patch = mpatches.Patch(color='red', label='Training')
    green_patch = mpatches.Patch(color='green', label='Test')
    plt.legend(handles=[red_patch, green_patch])

    plt.tight_layout()
    plt.show()
    return






def incorrectActions( ):
    file = "output/balabit_features_training.csv"
    data = pd.read_csv( file )
    n_from = data['n_from']
    n_to = data['n_to']
    userid = data['class']
    sessionid = data['session']
    n = len(n_from)
    print("LEN: "+str(n))
    for i in range(0,n):
        if int(n_from[i])>int(n_to[i]):
            print(str(userid[i])+","+str(sessionid[i])+","+str(n_from[i])+","+str(n_to[i]))
    return


def plotAucVsNumActionsSessionCut2():
    num_points = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

    # 1..20
    # auc_no     = np.array([0.7518, 0.7795, 0.7939, 0.8036, 0.8116, 0.8163,0.8220, 0.8253, 0.8273, 0.8310, 0.8317, 0.8345, 0.8376, 0.8365, 0.8374, 0.84000, 0.8423, 0.8403, 0.8423, 0.8442])
    # eer_no     = np.array([0.3263, 0.3025, 0.2924, 0.2769, 0.2749, 0.2736, 0.2626, 0.2628, 0.2584, 0.2535, 0.2557, 0.2496, 0.2500, 0.2568, 0.2497, 0.2459, 0.2391, 0.2475, 0.2423, 0.24000])

    auc_no = np.array([0.7518, 0.7795, 0.8036, 0.8163, 0.8253, 0.8310, 0.8345, 0.8365, 0.84000, 0.8403, 0.8442, 0.8478, 0.8482, 0.8443, 0.8492, 0.8485])
    eer_no = np.array([0.3263, 0.3025, 0.2769, 0.2736, 0.2628, 0.2535, 0.2496, 0.2568, 0.2459, 0.2475, 0.24000, 0.2401, 0.2262, 0.2397, 0.2469, 0.2432])
    # AUC and EER curves
    plt.plot(num_points, auc_no,'g', num_points, eer_no,'r')
    plt.axis([1, 30, 0.0, 0.9])
    plt.xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

    plt.title('Modeling unit: single action. AUC/EER  vs. number of actions')
    plt.xlabel('Number of actions')
    plt.ylabel('AUC/EER')
    plt.grid(linestyle='dashed', linewidth=0.5)
    green_patch   = mpatches.Patch(color='green', label='AUC')
    red_patch = mpatches.Patch(color='red', label='EER')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    return


def plotAucVsNumActionsSessionCut1():
    num_points = np.array([10,20,30,40,50,60,70,80,90,100])

    auc_no     = np.array([0.8297, 0.8601, 0.8531, 0.8538, 0.8522, 0.8453, 0.8427, 0.8387, 0.8437, 0.8402])
    eer_no     = np.array([0.2567, 0.2181, 0.2322, 0.2243, 0.2222, 0.2293, 0.2320, 0.2416, 0.2298, 0.2365 ])

    # AUC and EER curves
    plt.plot(num_points, auc_no,'g', num_points, eer_no,'r')
    plt.axis([10, 100, 0.0, 0.9])
    plt.xticks([10,20,30,40,50,60,70,80,90,100])

    plt.title('Modeling unit: sequence of actions. AUC/EER  vs. number of actions')
    plt.xlabel('Number of actions used for detection')
    plt.ylabel('AUC/EER')
    plt.grid(linestyle='dashed', linewidth=0.5)
    green_patch   = mpatches.Patch(color='green', label='AUC')
    red_patch = mpatches.Patch(color='red', label='EER')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    return


def plotEerVsNumPoints():
    num_points = np.array([4, 6, 8, 10, 12])
    eer_no = np.array([0.2031, 0.2039, 0.2000, 0.1958, 0.1964])
    eer_linear = np.array([0.2042, 0.2005, 0.2005, 0.1963, 0.1942])
    eer_spline = np.array([0.2326, 0.2265, 0.2434, 0.2359, 0.23535])

    # EER curves
    plt.plot(num_points, eer_no, 'r', num_points, eer_linear, 'g*', num_points, eer_spline, 'b')
    plt.axis([4, 12, 0.1, 0.3])
    plt.title('Equal Error Rate (EER)  vs. Number of points in actions')
    plt.xlabel('Number of points in actions')
    plt.ylabel('Equal Error Rate (EER)')
    red_patch = mpatches.Patch(color='red', label='No')
    green_patch = mpatches.Patch(color='green', label='Linear')
    blue_patch = mpatches.Patch(color='blue', label='Spline')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.show()
    return



def plotNumPointsHisto():
    input_training_file = "output/balabit_features_training.csv"
    input_test_file = "output/balabit_features_test.csv"
    training = pd.read_csv(input_training_file)
    test = pd.read_csv(input_test_file)
    numpoints_training = training['num_points']
    numpoints_test = test['num_points']

    titleStr = 'Number of points (mouse events) in actions'
    xlabelStr = 'Number of points'
    ylabelStr = 'Frequency'
    bsp.plotHistogramsTogether(numpoints_training, numpoints_test, titleStr, xlabelStr, ylabelStr)
    return



