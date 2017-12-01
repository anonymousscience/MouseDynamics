import csv
import os
import numpy as np


# used for SESSION_CUT = 2
# filename: balabit_features_test.csv  OR balabit_features_training.csv
def session_action_statistics( filename):
    print('**********')
    print('FILE: '+filename +" - session actions - statistics")
    file = open(filename, "r")
    reader = csv.DictReader(file)
    prevSessionid=""
    # prevUserid = ""
    MM = 0
    PC = 0
    DD = 0
    numMM = []
    numPC = []
    numDD = []
    numALL = []
    for row in reader:
        # type_of_action = row["type_of_action"]
        sessionid = row["session"]
        # userid = row["class"]
        typeOfAction = row["type_of_action"]
        if sessionid != prevSessionid:
            if prevSessionid != "":
                # print(prevSessionid+","+str(MM)+","+str(PC)+","+str(DD))
                numMM.append(MM)
                numPC.append(PC)
                numDD.append(DD)
                numALL.append(MM+PC+DD)

                MM = 0
                PC = 0
                DD = 0
        if( typeOfAction == "1"):
            MM = MM + 1
        if (typeOfAction == "3"):
            PC = PC + 1
        if (typeOfAction == "4"):
            DD = DD + 1
        prevSessionid = sessionid

    # print(sessionid + "," + str(MM) + "," + str(PC) + "," + str(DD))
    numMM.append(MM)
    numPC.append(PC)
    numDD.append(DD)
    numALL.append(MM + PC + DD)

    print('Table actions/sessions statistics')
    print('actiontype,mean,std,min,max')
    arrayMM = np.asarray(numMM)
    arrayPC = np.asarray(numPC)
    arrayDD = np.asanyarray(numDD)
    arrayALL = np.asanyarray(numALL)
    print ('MM,'+ str(np.mean(arrayMM))+","+str (np.std(arrayMM))+","+str(np.min(arrayMM))+","+str(np.max(arrayMM)) )
    print ('PC,'+ str(np.mean(arrayPC))+","+str (np.std(arrayPC))+","+str(np.min(arrayPC))+","+str(np.max(arrayPC)) )
    print ('DD,'+ str(np.mean(arrayDD))+","+str (np.std(arrayDD))+","+str(np.min(arrayDD))+","+str(np.max(arrayDD)) )
    print ('ALL,'+ str(np.mean(arrayALL))+","+str (np.std(arrayALL))+","+str(np.min(arrayALL))+","+str(np.max(arrayALL)) )

    return

# used for SESSION_CUT = 2
# filename: balabit_features_test.csv  OR balabit_features_training.csv
def user_action_statistics( filename ):
    print(filename + " - user actions - statistics")
    file = open(filename, "r")
    reader = csv.DictReader(file)
    prevUserid = ""
    MM = 0
    PC = 0
    DD = 0
    for row in reader:
        # type_of_action = row["type_of_action"]
        userid = row["class"]
        typeOfAction = row["type_of_action"]
        if userid != prevUserid:
            if prevUserid != "":
                print(prevUserid+","+str(MM)+","+str(PC)+","+str(DD))
                MM = 0
                PC = 0
                DD = 0
        if( typeOfAction == "1"):
            MM = MM + 1
        if (typeOfAction == "3"):
            PC = PC + 1
        if (typeOfAction == "4"):
            DD = DD + 1

        prevUserid = userid
    print(userid + ","  + str(MM) + "," + str(PC) + "," + str(DD))
    return



# used for SESSION_CUT = 2, case = 'training'
def training_time_statistics( training_filename ):
    file = open( training_filename, "r" )
    reader = csv.DictReader(file)
    dict = {}
    for row in reader:
        # type_of_action = row["type_of_action"]
        elapsed_time = float(row["elapsed_time"])
        userid = row["class"]
        if userid in dict.keys():
            dict[userid] = dict[userid] + elapsed_time
        else:
            dict[ userid ] = elapsed_time
    for i in dict:
        print(str(i) + ":" + str(dict[i]/3600) )
    return


# used for SESSION_CUT = 2, case = 'training'
def training_action_statistics( training_filename ):
    file = open( training_filename, "r" )
    reader = csv.DictReader(file)
    dict = {}
    for row in reader:
        # type_of_action = row["type_of_action"]
        elapsed_time = float(row["elapsed_time"])
        userid = row["class"]
        if userid in dict.keys():
            dict[userid] = dict[userid] + elapsed_time
        else:
            dict[ userid ] = elapsed_time
    for i in dict:
        print(str(i) + ":" + str(dict[i]/3600) )
    return


# used for SESSION_CUT = 2, case = 'training'
def test_time_statistics( test_filename ):
    file = open( test_filename, "r" )
    reader = csv.DictReader(file)
    dictlegal = {}
    dictilegal = {}
    for row in reader:
        # type_of_action = row["type_of_action"]

        elapsed_time = float(row["elapsed_time"])
        userid = row["class"]
        islegal = row[" islegal"]

        if islegal == "0":
            if userid in dictilegal.keys():
                dictilegal[userid] = dictilegal[userid] + elapsed_time
            else:
                dictilegal[ userid ] = elapsed_time
        else:
            if userid in dictlegal.keys():
                dictlegal[userid] = dictlegal[userid] + elapsed_time
            else:
                dictlegal[userid] = elapsed_time
    for i in dictlegal:
        print(str(i) + ":" + str(dictlegal[i]/3600) + ":" + str(dictilegal[i]/3600))
    return

# used for SESSION_CUT = 2
def session_time_statistics( filename ):
    print(filename + " - session time - statistics")
    file = open(filename, "r")
    reader = csv.DictReader(file)
    prevSessionid = ""
    prevUserid = ""
    sessionDuration = 0
    for row in reader:
        # type_of_action = row["type_of_action"]
        sessionid = row["session"]
        userid = row["class"]
        typeOfAction = row["type_of_action"]
        actionDuration = float(row["elapsed_time"])
        if sessionid != prevSessionid:
            if prevSessionid != "":
                print(prevUserid + "," + prevSessionid + "," + str(sessionDuration/60))
                sessionDuration = 0
        sessionDuration = sessionDuration + actionDuration
        prevSessionid = sessionid
        prevUserid = userid

    print(userid + "," + sessionid + "," + str(sessionDuration / 60))
    return

# used for SESSION_CUT = 2
def session_direction_statistics( filename ):
    print(filename + " - session direction - statistics")
    file = open(filename, "r")
    reader = csv.DictReader(file)

    dict = {'0':0, '1':0,'2':0, '3':0, '4':0, '5':0, '6':0, '7':0}

    prevSessionid = ""
    prevUserid = ""
    for row in reader:
        # type_of_action = row["type_of_action"]
        sessionid = row["session"]
        userid = row["class"]
        direction = row["direction_of_movement"]
        if sessionid != prevSessionid:
            if prevSessionid != "":
                outStr = ""
                for i in dict:
                    outStr = outStr + str(dict[i]) + ","
                print(prevUserid + "," + prevSessionid + "," + outStr)
                dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
        dict[direction] = dict[direction] + 1
        prevSessionid = sessionid
        prevUserid = userid
    outStr = ""
    for i in dict:
        outStr = outStr + str(dict[i]) + ","
    print(userid + "," + sessionid + "," + outStr)

    return

# returns 0 if the session does not contain Scoll events
# otherwise returns 1
def containsScrolls( filename ):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['button'] == 'Scroll':
                return 1
    return 0

# statistics
def rawDataStatistics( filename ):
    counter = 1
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['button'] == 'Right':
                counter = counter + 1

    return counter

def session_rightClick_statistics( filename ):
    file = open(filename, "r")
    reader = csv.DictReader(file)
    counter = 0
    for row in reader:
        if row["button"] == 'Right':
            counter = counter + 1
    return counter


# used for SESSION_CUT = 2
def print_session_numActions( filename ):
    print(filename + " - session num actions - statistics")
    file = open(filename, "r")
    reader = csv.DictReader(file)
    prevSessionid = ""
    numActions = 0
    for row in reader:
        sessionid = row["session"]
        if sessionid != prevSessionid:
            if prevSessionid != "":
                print(prevSessionid + "," + str(numActions))
                numActions = 0
        numActions += 1
        prevSessionid = sessionid
    print(prevSessionid + "," + str(numActions))
    return

# case: train, test
# folder: folder containing the users's directories
# dlabels: return value of the main.process_public_labels()

def count_sessions ( case, folder, dlabels ):
    # for each user the number of legal and illegal sesions are counted
    directory = os.fsencode( folder )
    dictlegal = {}
    dictilegal = {}
    counter = 0
    print("userid,#sessions")
    for fdir in os.listdir(directory):
        dirname = os.fsdecode(fdir)
        userdirectory = folder + '/' + dirname
        # print(userdirectory)
        # is_legal is not used in case of training
        is_legal = 0
        userid = dirname[4:len(dirname)]
        numLegals = 0
        numIllegals = 0
        numSessions = 0
        for file in os.listdir(userdirectory):
            fname = os.fsdecode(file)
            filename = userdirectory + '/' + os.fsdecode(file)
            sessionid = fname[8:len(fname)]
            counter += 1
            # nem minden teszfajlnak ismert a cimkeje
            if case == 'test' and not sessionid in dlabels:
                continue
            # print('File: ' + fname)

            if case == 'test':
                is_legal = dlabels[sessionid]
                if is_legal == 1:
                    numLegals = numLegals + 1
                else:
                    numIllegals = numIllegals + 1
                # print(is_legal)
            else:
                numSessions = numSessions + 1
        if case == 'test':
            print(str(userid)+","+str(numLegals)+","+str(numIllegals))
        else:
            print(str(userid) + "," + str(numSessions))

    print("Num session files: " + str(counter))
    return


# counts the number of sessions containing scroll mouse events
# case: train, test
# folder: folder containing the users's directories
# dlabels: return value of the main.process_public_labels()

def countScrolls( case, folder, dlabels ):
    numSessions = 0
    scrollSessions = 0
    directory = os.fsencode(folder )
    # for each user
    for fdir in os.listdir(directory):
        dirname = os.fsdecode(fdir)
        userdirectory = folder + '/' + dirname
        userid = dirname[4:len(dirname)]
        for file in os.listdir(userdirectory):
            fname = os.fsdecode(file)
            filename = userdirectory + '/' + os.fsdecode(file)
            sessionid = fname[8:len(fname)]

            # nem minden teszfajlnak ismert a cimkeje
            if case == 'test' and not sessionid in dlabels:
                continue
            numSessions = numSessions + 1
            result = containsScrolls(filename)
            if result == 1:
                scrollSessions = scrollSessions + 1
    print("Case "+case+", total number of sessions: " + str(numSessions))
    print("Num session files containing scrool events: " + str(scrollSessions))
    return


