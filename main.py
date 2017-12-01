import plot_paper
import balabit_statistics as bs
import plot_actions_session
import os

import settings as st
import twoclass_classification as tc
import user_accuracies as ua
import rawdata2actions as rd
import feature_selection as fs
import plot_paper as pp

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# SESSION_CUT = 1
# modeling unit: SEQUENCE of ACTIONS
def main_ACTION_SEQUENCE():
    st.SESSION_CUT = 1
    st.CASE = 'training'
    print("***Computing training features")
    rd.process_files(st.CASE)
    print('***Evaluating on the test set')
    st.CASE = 'test'
    rd.process_files(st.CASE)
    tc.NUM_NEGATIVE_SAMPLES_PER_CLASS = 70
    tc.NUM_POSITIVE_SAMPLES = 630
    # in case of this modeling unit (sequence of actions), from each test session we extract exactly one feature vector
    tc.evaluate_test_actions2(st.TRAINING_FEATURE_FILENAME, st.TEST_FEATURE_FILENAME)
    return


# SESSION_CUT = 2
# modeling unit: ACTION
def main_ACTION():
    st.SESSION_CUT = 2
    print("***Computing training features")
    st.CASE = 'training'
    rd.process_files( st.CASE )
    print('***Evaluating on the test set')
    st.CASE = 'test'
    rd.process_files(st.CASE)
    print('EVAL_TEST_UNIT: '+str(st.EVAL_TEST_UNIT))
    tc.NUM_NEGATIVE_SAMPLES_PER_CLASS = 200
    tc.NUM_POSITIVE_SAMPLES = 1800
    if st.EVAL_TEST_UNIT == 0:
        # evaluation: all actions from a session
        tc.evaluate_test_session(st.TRAINING_FEATURE_FILENAME, st.TEST_FEATURE_FILENAME)
    else:
        # evaluation: action by action
        tc.evaluate_test_actions(st.TRAINING_FEATURE_FILENAME, st.TEST_FEATURE_FILENAME,
                                                      st.NUM_EVAL_ACTIONS)
    return


# SESSION_CUT = 2
# modeling unit: ACTION
def main_ACTION2(NUMACTIONS):
    st.SESSION_CUT = 2
    print("***Computing training features")
    st.CASE = 'training'
    rd.process_files( st.CASE )
    print('***Evaluating on the test set')
    st.CASE = 'test'
    rd.process_files(st.CASE)
    print('EVAL_TEST_UNIT: '+str(st.EVAL_TEST_UNIT))
    tc.NUM_NEGATIVE_SAMPLES_PER_CLASS = 200
    tc.NUM_POSITIVE_SAMPLES = 1800
    if st.EVAL_TEST_UNIT == 0:
        # evaluates only the test sessions having at least NUMACTIONS actions
        tc.evaluate_test_session_having_at_least(st.TRAINING_FEATURE_FILENAME, st.TEST_FEATURE_FILENAME, NUMACTIONS)
        # evaluation: all actions from a session
        # tc.evaluate_test_session(st.TRAINING_FEATURE_FILENAME, st.TEST_FEATURE_FILENAME)
    else:
        # evaluation: action by action
        tc.evaluate_test_actions(st.TRAINING_FEATURE_FILENAME, st.TEST_FEATURE_FILENAME,
                                                      st.NUM_EVAL_ACTIONS)
    return


# plots all sessions from a folder
# @param case: training OR test
# @param toSave:
#        True: plots are not shown but saved in PNG and EPS formats
#        False: plots are shown on the display
def plot_all ( case, toSave ):
    if case == 'training':
        feature_filename = st.TRAINING_FEATURE_FILENAME
    else:
        feature_filename = st.TEST_FEATURE_FILENAME

    if case == 'test':
        directory = os.fsencode(st.BASE_FOLDER + st.TEST_FOLDER)
    else:
        directory = os.fsencode(st.BASE_FOLDER + st.TRAINING_FOLDER)
    for fdir in os.listdir(directory):
        dirname = os.fsdecode(fdir)
        print('User: ' + dirname)
        if case == 'test':
            userdirectory = st.BASE_FOLDER + st.TEST_FOLDER + '/' + dirname
        else:
            userdirectory = st.BASE_FOLDER + st.TRAINING_FOLDER + '/' + dirname
        # is_legal is not used in case of training
        is_legal = 0
        userid = dirname[4:len(dirname)]
        for file in os.listdir(userdirectory):
            fname = os.fsdecode(file)
            filename = userdirectory + '/' + os.fsdecode(file)
            sessionid = str(fname[8:len(fname)])
            print(dirname+" : "+sessionid)
            plot_actions_session.plot_all_actions_sesssion(feature_filename, dirname, sessionid, toSave)

    return




# main_ACTION_SEQUENCE()
main_ACTION()




