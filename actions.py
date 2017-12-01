import numpy as np
import math

import general_statistics as gs
import settings as st
import time_sample_interpolation as ts


# computes features from an action
# input:
# returns a tuple representing a feature vector
def computeActionFeatures(x, y, t, actionCode, action_file, n_from, n_to):
    n = len(x)
    if st.GLOBAL_DEBUG:
        print("\t\t"+str(n_from)+'-'+str(n_to)+" len: "+str(n))

    # short actions are filtered out
    if n < st.GLOBAL_MIN_ACTION_LENGTH:
        return None
    x = [int(e) for e in x]
    y = [int(e) for e in y]
    t = [float(e) for e in t]

    if gs.containsNull( x ):
        # Scroll action
        return None

    for i in range(1, n):
        if (x[i] > st.X_LIMIT or y[i] > st.Y_LIMIT):
            x[i] = x[i - 1]
            y[i] = y[i - 1]

    # lastNotNullIndex = 0
    # for i in range (1,n):
    #     if (x[i] > st.X_LIMIT or y[i] > st.Y_LIMIT):
    #         x[i] = x[i - 1]
    #         y[i] = y[i - 1]
    #     if x[i]!= 0 and y[i] != 0:
    #         lastNotNullIndex = i
    #     if x[i] == 0 and y[i] == 0 and lastNotNullIndex != 0:
    #         x[i] = x[lastNotNullIndex]
    #         y[i] = y[lastNotNullIndex]


    if st.INTERPOLATION_TYPE != 'NO':
        # ********************************
        # call interpolation function
        result = None
        if st.INTERPOLATION_TYPE == 'LINEAR':
            xyt_line_array = np.column_stack((np.array(x), np.array(y), np.array(t)))
            result = ts.timeSampleInterpolationLinear(xyt_line_array, st.FREQUENCY)
        else:
            if st.INTERPOLATION_TYPE == 'POLINOMIAL':
                xyt_line_array = np.column_stack((np.array(x), np.array(y), np.array(t)))
                result = ts.timeSampleInterpolationPolinomial(xyt_line_array, st.FREQUENCY)
            else:
                if st.INTERPOLATION_TYPE == 'SPLINE':
                    xyt_line_array = np.column_stack((np.array(x), np.array(y), np.array(t)))
                    result = ts.timeSampleInterpolationSpline(xyt_line_array, st.FREQUENCY)
            if result != None:
                # print('\t'+str(xyt_line_array.shape)+" : "+ str(result.shape))
                x = result[:, 0]
                y = result[:, 1]
                t = result[:, 2]
                # print(str(n)+" : "+str(len(x)))
                n = len(x)
            else:
                print('Interpolation error. No interpolation.')
                print('\t'+str(n_from) + '-' + str(n_to) + " len: " + str(n))
        # ********************************

    # trajectory from the beginning point of the action
    # angles
    trajectory = 0
    sumOfAngles = 0
    angles = [ 0]
    path = [ 0 ]
    # velocities
    vx = [ 0 ]
    vy = [ 0]
    v = [ 0 ]
    for i in range(1, n):
        dx = int(x[i]) - int(x[i - 1])
        dy = int(y[i]) - int(y[i - 1])
        dt = float(t[i]) - float(t[i-1])
        if dt == 0:
            dt = 0.01
        vx_val = dx/dt
        vy_val = dy/dt
        vx.append(vx_val)
        vy.append(vy_val)
        v.append(math.sqrt(vx_val * vx_val +vy_val* vy_val))
        angle = math.atan2(dy,dx)
        angles.append( angle )
        sumOfAngles += angle
        distance = math.sqrt( dx * dx + dy *dy)
        trajectory = trajectory + distance
        path.append(trajectory)

    mean_vx = gs.mean(vx, 0, len(vx))
    sd_vx = gs.stdev(vx, 0, len(vx))
    max_vx = gs.max(vx, 0, len(vx))

    mean_vy = gs.mean(vy, 0, len(vy))
    sd_vy = gs.stdev(vy, 0, len(vy))
    max_vy = gs.max(vy, 0, len(vy))

    mean_v = gs.mean(v, 0, len(v))
    sd_v = gs.stdev(v, 0, len(v))
    max_v = gs.max(vy, 0, len(v))

    # angular velocity
    omega = [0]
    no = len(angles)
    for i in range(1, no):
        dtheta = angles[ i ]-angles[ i-1 ]
        dt = float(t[i]) - float(t[i - 1])
        if dt == 0:
            dt = 0.01
        omega.append( dtheta/dt)

    mean_omega = gs.mean(omega, 0, len(omega))
    sd_omega = gs.stdev(omega, 0, len(omega))
    max_omega = gs.max(omega, 0, len(omega))

    # acceleration
    a = [ 0 ]
    accTimeAtBeginning = 0
    cont = True
    for i in range(1, n - 1):
        dv = v[ i ] - v[ i-1 ]
        dt = float(t[i]) - float(t[i - 1])
        if dt == 0:
            dt = 0.01
        if cont  and dv > 0 :
            accTimeAtBeginning += dt
        else:
            cont = False
        a.append( dv/dt)
    mean_a = gs.mean(a, 0, len(a) )
    sd_a = gs.stdev(a, 0, len(a) )
    max_a = gs.max(a, 0, len(a) )

    # jerk
    j = [0]
    na = len(a)
    for i in range(1, na):
        da = a[i] - a[i - 1]
        dt = float(t[i]) - float(t[i - 1])
        if dt == 0:
            dt = 0.01
        j.append(da / dt)
    mean_jerk = gs.mean(j, 0, len(j))
    sd_jerk = gs.stdev(j, 0, len(j))
    max_jerk = gs.max(j, 0, len(j))


    # curvature: defined by Gamboa&Fred, 2004
    # numCriticalPoints
    c = []
    numCriticalPoints = 0
    nn = len(path)
    for i in range(1, nn):
        dp = path[i]-path[i-1]
        if dp == 0:
            continue
        dangle = angles[i] - angles[i - 1]
        curv = dangle/dp
        c.append( curv )
        if abs(curv) < st.CURV_THRESHOLD:
            numCriticalPoints = numCriticalPoints + 1
    mean_curv = gs.mean( c, 0, len(c))
    sd_curv = gs.stdev(c, 0, len(c))
    max_curv = gs.max(c, 0, len(c))

    # time
    time = float(t[n - 1]) - float(t[0])

    # direction: -pi..pi
    theta = math.atan2(int(y[n - 1]) - int(y[0]), int(x[n - 1]) - int(x[0]))
    direction = computeDirection(theta)

    # distEndtToEndLine
    distEndToEndLine = math.sqrt((int(x[0]) -int(x[n-1])) * (int(x[0]) -int(x[n-1])) +(int(y[0]) -int(y[n-1]))*(int(y[0]) -int(y[n-1])))

    # straightness
    if trajectory == 0:
        straightness = 0
    else:
        straightness =  distEndToEndLine / trajectory
    if straightness > 1:
        straightness = 1

    # largest deviation
    largest_deviation = largestDeviation(x,y)


    return str(actionCode) + ',' + str(trajectory) + ',' + str(time) + ',' + str(direction) + ','+\
                      str(straightness)+ ','+ str(n)+','+str(sumOfAngles)+','+\
                      str(mean_curv) + "," + str(sd_curv) + "," + str(max_curv) + "," +\
                      str(mean_omega)+","+str(sd_omega)+","+str(max_omega)+","+\
                      str(largest_deviation)+","+\
                      str(distEndToEndLine)+","+str(numCriticalPoints)+","+\
                      str(mean_vx)   + "," + str(sd_vx)   + "," +str(max_vx) + "," +\
                      str(mean_vy)   + "," + str(sd_vy)   + "," +str(max_vy) + "," +\
                      str(mean_v)    + "," + str(sd_v)    + "," +str(max_v)  + "," +\
                      str(mean_a)    + "," + str(sd_a)    + "," + str(max_a) + "," +\
                      str(mean_jerk) + "," + str(sd_jerk) + "," + str(max_jerk) + "," +\
                      str(accTimeAtBeginning)+","+\
                      str(n_from)+","+str(n_to)+\
                      "\n"

# computer features from an action [n_from, n_to]
# and writes into the action file
def printAction(x, y, t, actionCode, action_file, n_from, n_to):
    result = computeActionFeatures(x, y, t, actionCode, action_file, n_from, n_to)
    if result != None:
        action_file.write(result)
    return


def largestDeviation(x, y):
    n = len( x )
    #     line (x_0,y_0) and (x_n-1,y_n-1): ax + by + c
    a = float(x[n-1]) - float(x[0])
    b = float(y[0]) - float(y[n-1])
    c = float(x[0]) * float(y[n-1]) - float(x[n-1]) * float(y[0])
    max = 0
    den = math.sqrt(a*a+b*b)
    for i in range(1,n-1):
    #     distance x_i,y_i from line
        d = math.fabs(a*float(x[i])+b*float(y[i])+c)
        if d > max:
            max = d
    if den > 0:
        max /= den
    return max


# one PC action
def processPointClick(x, y, t, action_file, n_from, n_to):
    printAction(x, y, t, st.PC, action_file, n_from, n_to)
    return

# one MM action
def processMouseMove(x, y, t, action_file, n_from, n_to):
    printAction(x, y, t, st.MM, action_file, n_from, n_to)
    return

# one DD action
def processDragAction(x, y, t, action_file, n_from, n_to):
    printAction(x, y, t, st.DD, action_file, n_from, n_to)
    return


# can be a compound action: {MM}*PC
def processPointClickActions(data, action_file, n_from, n_to):
    if st.GLOBAL_DEBUG:
        print("MM*PC:" + str(n_from) + "-" + str(n_to))
    x = []
    y = []
    t = []
    prevTime = 0
    start = n_from
    counter = 0
    for item in data:
        actState = item['state']
        actTime = float(item['t'])
        counter += 1
        if actState == 'Pressed':
            if len(t) > st.GLOBAL_MIN_ACTION_LENGTH:
                x.append(item['x'])
                y.append(item['y'])
                t.append(actTime)
                if st.GLOBAL_DEBUG:
                    print("\tPC:"+str(start)+"-"+str(n_to))
                processPointClick(x, y, t, action_file, start, n_to)
            return
        else:
            # idokuszob miatt vagunk
            if actTime - prevTime > st.GLOBAL_DELTA_TIME:
                stop = n_from + counter - 2
                # ha eleg hosszu az adatsor mentjuk
                if len(t) > st.GLOBAL_MIN_ACTION_LENGTH:
                    if st.GLOBAL_DEBUG:
                        print("\tMM:" + str(start) + "-" + str(stop))
                    processMouseMove(x, y, t, action_file, start, stop)
                # uj AcTION kezdodik
                x = []
                y = []
                t = []
                start = stop+1
            else:
                x.append(item['x'])
                y.append(item['y'])
                t.append(actTime)
        prevTime = actTime
    return

# can be a compound action: {MM}*
def processMouseMoveActions(data, action_file, n_from, n_to):
    if st.GLOBAL_DEBUG:
        print("MM*MM:" + str(n_from) + "-" + str(n_to))
    x = []
    y = []
    t = []
    start = n_from
    counter = 0
    prevTime = 0
    for item in data:
        x.append(item['x'])
        y.append(item['y'])
        counter += 1
        actTime = float(item['t'])
        t.append(actTime)
        if actTime - prevTime > st.GLOBAL_DELTA_TIME:
            stop = n_from + counter - 2
            if len(t) > st.GLOBAL_MIN_ACTION_LENGTH:

                if st.GLOBAL_DEBUG:
                    print("\tMM:" + str(start) + "-" + str(stop))
                processMouseMove(x, y, t, action_file, start, stop)
                x = []
                y = []
                t = []
                start = stop+1
        prevTime = actTime
    if len(t) > st.GLOBAL_MIN_ACTION_LENGTH:
        if st.GLOBAL_DEBUG:
            print("\tMM:" + str(start) + "-" + str(n_to))
        processMouseMove(x, y, t, action_file, start, n_to)
    return

# {MM}*DD
def processDragActions(data, action_file, n_from, n_to):
    if st.GLOBAL_DEBUG:
        print("MM*DD:" + str(n_from) + "-" + str(n_to))
    x = []
    y = []
    t = []
    start = n_from
    stop = start
    counter = 0
    prevTime = 0
    for item in data:
        actButton = item['button']
        actState = item['state']
        actTime = float(item['t'])
        counter += 1
        if actButton == 'NoButton' and actState == 'Move':
            if actTime - prevTime > st.GLOBAL_DELTA_TIME:
                stop = n_from + counter - 2
                if len(t) > st.GLOBAL_MIN_ACTION_LENGTH:
                    if st.GLOBAL_DEBUG:
                        print("\tMM:" + str(start) + "-" + str(stop))
                    processMouseMove(x, y, t, action_file, start, stop)
                x = []
                y = []
                t = []
                start = stop +1
            x.append(item['x'])
            y.append(item['y'])
            t.append(actTime)

        if actButton == 'Left' and actState == 'Pressed':
            # ends the MM action
            if len(t) > st.GLOBAL_MIN_ACTION_LENGTH:
                stop = n_from + counter - 2
                if st.GLOBAL_DEBUG:
                    print("\tMM:" + str(start) + "-" + str(stop))
                processMouseMove(x, y, t, action_file, start, stop)
            # starts the DD action
            x = []
            y = []
            t = []
            start = stop +1
            x.append(item['x'])
            y.append(item['y'])
            t.append(actTime)
        if actButton == 'Left' and actState == 'Released':
            # ends the DD action
            x.append(item['x'])
            y.append(item['y'])
            t.append(actTime)
            if st.GLOBAL_DEBUG:
                print("\tDD:" + str(start) + "-" + str(n_to))
            processDragAction(x, y, t, action_file, start, n_to)
        if actButton == 'NoButton' and actState == 'Drag':
            x.append(item['x'])
            y.append(item['y'])
            t.append(actTime)
        prevTime = actTime
    return

# directions: 0..7
# Ahmed & Traore, IEEE TDSC2007
def computeDirection(theta):
    direction = 0
    if 0 <= theta < math.pi / 4:
        direction = 0
    if math.pi / 4 <= theta < math.pi / 2:
        direction = 1
    if math.pi / 2 <= theta < 3 * math.pi / 4:
        direction = 2
    if 3 * math.pi / 4 <= theta < math.pi:
        direction = 3
    if -math.pi / 4 <= theta < 0:
        direction = 7;
    if -math.pi / 2 <= theta < -math.pi / 4:
        direction = 6;
    if -3 * math.pi / 4 <= theta < -math.pi / 2:
        direction = 5;
    if -math.pi <= theta < -3 * math.pi / 4:
        direction = 4;
    return direction


# directions: 0..7
# Chao Shen, TIFS, 2013-as cikk alapjan
# def computeDirection(theta):
#     direction = 0
#
#     if -math.pi / 8 <= theta < 0:
#         direction = 0
#     if 0 <= theta < math.pi / 8:
#         direction = 0
#
#
#     if math.pi/8 <= theta < 3*math.pi / 8:
#         direction = 1
#
#     if 3*math.pi / 8 <= theta < 5*math.pi / 8:
#         direction = 2
#
#     if 5*math.pi / 8 <= theta < 7 * math.pi / 8:
#         direction = 3
#
#     if 7*math.pi / 8 <= theta < math.pi:
#         direction = 4
#     if -math.pi <= theta <-7* math.pi/8:
#         direction = 4
#
#     if -7 * math.pi / 8 <= theta < -5*math.pi/8:
#         direction = 5
#
#     if -5 *math.pi / 8 <= theta < -3 *math.pi / 8:
#         direction = 6;
#
#     if -3 * math.pi / 8 <= theta < -math.pi / 8:
#         direction = 7;
#
#
#     return direction
