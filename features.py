import numpy


# used by SESSION_CUT = 1
# Mainly histogram based features
# Extracted from a sequence of actions havind the indexes in the feature file start and stop:  [start, stop)


def computeActionTypeFrequency(tdir, start, stop):
    # print(tdir[start:stop])
    fr = {'1': 0, '3': 0, '4': 0}
    for i in range(start, stop):
        if tdir[i] == 1:
            fr['1'] += 1
        if tdir[i] == 3:
            fr['3'] += 1
        if tdir[i] == 4:
            fr['4'] += 1
    # normalize
    n = stop - start
    fr['1'] /= n
    fr['3'] /= n
    fr['4'] /= n
    # print(str(fr['1'])+","+str(fr['3'])+","+str(fr['4']))
    return fr


def computeDirectionFrequency(direction, start, stop):
    fr = [0] * 8
    for i in range(start, stop ):
        fr[int(direction[i])] += 1
    n = stop - start
    fr[:] = [x / n for x in fr]
    return fr


def computeAverageTimePerActionType(actiontype, time, start, stop):
    avg_time = [0] * 5
    counter = [0] * 5

    for i in range(start, stop ):
        type = int(actiontype[i])
        avg_time[type] = avg_time[type] + float(time[i])
        counter[type] += 1

    for i in range(1, 5):
        if counter[i] != 0:
            avg_time[i] = avg_time[i] / counter[i]
    return avg_time


def computeTraveledDistanceHistogram(distance, start, stop):
    # 10 ranges: 0 - 100; 100 - 200;..;800 - 900;>=900
    fr = [0] * 10
    for i in range(start, stop ):
        d = float(distance[i])
        if d >= 900:
            fr[9] += 1
        else:
            fr[(int)(d / 100)] += 1
    n = stop - start
    fr[:] = [x / n for x in fr]
    return fr


def computeVelocityHistogram(time, distance, numBins, start, stop):
    velocity = []
    for i in range(start, stop ):
        if float(time[i]) < 0.00001:
            continue
        vel = float(distance[i]) / float(time[i])
        velocity.append(vel)
    return histogram(velocity, numBins)



def computeSignalHistogram(signal, bins, start, stop):
    nsignal = []
    for i in range(start, stop ):
        nsignal.append(float(signal[i]))
    return histogram(nsignal, bins)

def histogram(signal, min, max, bins):
    distribution = [0] * bins
    step = (max - min) / bins;
    for i in range(0, len(signal)):

        for j in range(bins - 1, -1,-1):
            if signal[i] > min + j * step:
                distribution[j] += 1
                break;
    for i in range(0, bins ):
        distribution[i] /= len(signal)
    return distribution


def histogram(signal, bins):
    distribution = [0] * bins
    if len(signal) == 0:
        return distribution
    min = numpy.min(signal)
    max = numpy.max(signal)
    step = (max - min) / bins;
    counter = 0
    for i in range(0, len(signal)):

        for j in range(bins - 1, -1, -1):
            if signal[i] > min + j * step:
                distribution[j] += 1
                break;
    for i in range(0, bins):
        distribution[i] /= len(signal)
    return distribution



def computeMaxVelocity(time, distance, start, stop):
    max_vel = 0
    for i in range(start, stop):
        if float(time[i]) < 0.00001:
            continue
        vel = float(distance[i]) / float(time[i])
        if vel > max_vel:
            max_vel = vel
    return max_vel



def computeAverageVelocityPerActionType(actiontype, time, distance, start, stop):
    avg_vel = [0] * 5
    counter = [0] * 5

    for i in range(start, stop ):
        type = int(actiontype[i])
        if float(time[i]) < 0.00001:
            continue
        avg_vel[type] = avg_vel[type] + float(distance[i]) / float(time[i])
        counter[type] += 1

    for i in range(1, 5):
        if counter[i] != 0:
            avg_vel[i] = avg_vel[i] / counter[i]
    return avg_vel

def computeAverageVelocityPerDirection(tdistance, ttime, tdirection, ttype, whichType, start, stop):
    avg_vel = [0] * 8
    counter = [0] * 8

    for i in range(start, stop ):
        if str(ttype[i]) == str(whichType):
            idir = int(tdirection[i])
            if float(ttime[i]) != 0:
                avg_vel[idir] = avg_vel[idir] + float(tdistance[i]) / float(ttime[i])
                counter[idir] += 1

    for i in range(0, 8):
        if counter[i] != 0:
            avg_vel[i] = avg_vel[i] / counter[i]
    return avg_vel