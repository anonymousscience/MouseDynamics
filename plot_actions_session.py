import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time_sample_interpolation as ts

import settings as st


# Plots the actions from a session
# Actions endpoints are taken from feature files
# Actions raw data are takn from the raw session file
def plot_sesssion(feature_file, userid, sessionid):
    print(feature_file)
    featdata = pd.read_csv(feature_file)
    session = featdata['session']
    type_of_action = featdata['type_of_action']
    n_from = featdata['n_from']
    n_to = featdata['n_to']
    numActions = len(session)
    print("Num. actions: "+str(numActions))

    if feature_file.find('test') != -1:
        path= st.BASE_FOLDER+st.TEST_FOLDER+'/'+userid+'/'
    else:
        path = st.BASE_FOLDER + st.TRAINING_FOLDER + '/'+userid+'/'

    session_filename = path +'session_'+sessionid
    print(session_filename)
    rawdata = pd.read_csv(session_filename)
    x = rawdata['x']
    y = rawdata['y']
    t = rawdata['client timestamp']
    print('Num. lines: '+ str(len(x)))
    actionCounter = 0
    # iterate through feature file
    while actionCounter < numActions:
        # print( int(session[actionCounter]) )
        if ( int(session[actionCounter]) != int(sessionid) ):
            actionCounter += 1
            continue
        start_index = n_from[actionCounter]
        stop_index = n_to[actionCounter]

        if stop_index - start_index < 1:
            continue
        if stop_index - start_index < st.GLOBAL_MIN_ACTION_LENGTH:
            actionCounter += 1
            print('SHORT ACTION')
            continue

        print('ACTION: ' + str(start_index) + " : " + str(stop_index))

        # xo, yo - original data
        xo = x[start_index:stop_index]
        yo = y[start_index:stop_index]
        to = t[start_index:stop_index]

        # print(xo)

        xo = [int(e) for e in xo]
        yo = [int(e) for e in yo]
        to = [float(e) for e in to]

        no = len(xo)


        # if gs.containsNull(xo):
        #     # Scroll action
        #     return None

        for i in range(1, no):
            if (xo[i] > st.X_LIMIT or yo[i] > st.Y_LIMIT):
                xo[i] = xo[i - 1]
                yo[i] = yo[i - 1]

        if type_of_action[actionCounter] == 1:
            title = 'Mouse Move: '
        else:
            if type_of_action[actionCounter] == 3:
                title = 'Point Click: '
            else:
                title = 'Drag and Drop: '
        title += str(start_index) + ' - ' + str(stop_index)

        # plot the original signal
        fig = plt.figure()
        plt.plot(xo, yo, '-', marker='*', markersize=6)
        plt.axis([0, 2100, 0, 1200])
        plt.title(title)
        plt.legend(['orig'], loc='best')


        # ********************************
        # call interpolation function
        result = None
        if st.INTERPOLATION_TYPE != 'NO':

            if st.INTERPOLATION_TYPE == 'LINEAR':
                xyt_line_array = np.column_stack((np.array(xo), np.array(yo), np.array(to)))
                result = ts.timeSampleInterpolationLinear(xyt_line_array, st.FREQUENCY)
            else:
                if st.INTERPOLATION_TYPE == 'POLINOMIAL':
                    xyt_line_array = np.column_stack((np.array(xo), np.array(yo), np.array(to)))
                    result = ts.timeSampleInterpolationPolinomial(xyt_line_array, st.FREQUENCY)
                else:
                    if st.INTERPOLATION_TYPE == 'SPLINE':
                        xyt_line_array = np.column_stack((np.array(xo), np.array(yo), np.array(to)))
                        result = ts.timeSampleInterpolationSpline(xyt_line_array, st.FREQUENCY)
            if result != None:
                xi = result[:, 0]
                yi = result[:, 1]
                ti = result[:, 2]
                ni = len(xi)
                print('\t\tOK\tno: '+str(no)+"\t ni: "+str(ni))
            else:
                print('\t\tINTERPOLATION ERROR:')

        # ********************************

        if type_of_action[actionCounter] == 1:
            title = 'Mouse Move: '
        else:
            if type_of_action[actionCounter] == 3:
                title = 'Point Click: '
            else:
                title = 'Drag and Drop: '
        title += str(start_index)+' - '+ str(stop_index)
        actionCounter += 1

        # print('\t\tPlot action: ' + title)
        # if st.INTERPOLATION_TYPE != 'NO' and result!= None :
        #     fig = plt.figure()
        #     plt.plot(xo, yo, '-', xi, yi, 'x')
        #     # plt.axis([0, 2500, 0, 1500])
        #     plt.title(title)
        #     plt.legend(['orig', 'interpolated'], loc='best')
        #     plt.show()
    plt.show()
    return

def plot_all_actions_sesssion(feature_file, userid, sessionid, toSave):
    print(feature_file)
    featdata = pd.read_csv(feature_file)
    session = featdata['session']
    type_of_action = featdata['type_of_action']
    n_from = featdata['n_from']
    n_to = featdata['n_to']
    numActions = len(session)
    # print("Num. actions in the feature file: "+str(numActions))

    if feature_file.find('test') != -1:
        path= st.BASE_FOLDER+st.TEST_FOLDER+'/'+userid+'/'
    else:
        path = st.BASE_FOLDER + st.TRAINING_FOLDER + '/'+userid+'/'

    session_filename = path +'session_'+sessionid
    print(session_filename)
    rawdata = pd.read_csv(session_filename)
    x = rawdata['x']
    y = rawdata['y']
    t = rawdata['client timestamp']
    print('Num. lines: '+ str(len(x)))
    actionCounter = 0

    fig = plt.figure()

    # iterate through feature file
    mm = mpatches.Patch(color='red', label='Mouse Move - MM')
    pc = mpatches.Patch(color='blue', label='Point Click - PC')
    dd = mpatches.Patch(color='green', label='Drag&Drop - DD')


    plt.legend(handles=[mm, pc, dd])
    plt.axis([0, 2100, 0, 1200])

    plotCounter = 0
    publicLabel = False
    while actionCounter < numActions:
        # print( int(session[actionCounter]) )
        if ( int(session[actionCounter]) != int(sessionid) ):
            actionCounter += 1
            continue
        publicLabel = True
        start_index = n_from[actionCounter]
        stop_index = n_to[actionCounter]

        if stop_index - start_index < 1:
            continue
        if stop_index - start_index < st.GLOBAL_MIN_ACTION_LENGTH:
            actionCounter += 1
            print('SHORT ACTION')
            continue

        # print('ACTION: ' + str(start_index) + " : " + str(stop_index))
        # xo, yo - original data
        xo = x[start_index:stop_index]
        yo = y[start_index:stop_index]
        to = t[start_index:stop_index]



        xo = [int(e) for e in xo]
        yo = [int(e) for e in yo]
        to = [float(e) for e in to]

        no = len(xo)

        for i in range(1, no):
            if (xo[i] > st.X_LIMIT or yo[i] > st.Y_LIMIT):
                xo[i] = xo[i - 1]
                yo[i] = yo[i - 1]


        if not containsZeros(xo, yo):
            if type_of_action[actionCounter] == 1:
                # MM - Mouse Move
                plt.plot(xo, yo, linestyle='-', color='red', marker='*', markersize=1.5, linewidth=0.5)
            else:
                if type_of_action[actionCounter] == 3:
                    # PC - Point Click
                    plt.plot(xo, yo, linestyle='-', color='blue', marker='*', markersize=1.5, linewidth=0.5)
                else:
                    # DD - Drag and Drop
                    plt.plot(xo, yo, linestyle='-', color='green', marker='*', markersize=1.5, linewidth=0.5)

                # plt.legend(['orig'], loc='best')
            plotCounter += 1
        actionCounter+=1

    if publicLabel:
        title = str(userid) + ', session: ' + str(sessionid)+", numActions: "+str(plotCounter)
        print(title+" "+str(plotCounter))
        plt.title(title)
        if toSave == True:
            plt.savefig(userid + "_" + sessionid + '.png')
            plt.savefig(userid + "_" + sessionid + '.eps')
        else:
            plt.show()
    else:
        print('session: ' + str(sessionid)+" is a private test file")
    return

def containsZeros( x, y):
    n = len(x)
    for i in range(1, n):
        if (x[i] == 0 or y[i] == 0 ):
            return True
    return False