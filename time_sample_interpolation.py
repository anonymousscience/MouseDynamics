import numpy as np
import scipy.interpolate as interp

"""
Input : an array with point coordinates (x, y) first two columns and the timestamps for each point. Typically sampled mouse movements
The array has to contain at least 4 points! (interp1d constraint). The function completes this sample with points and timestamps using
cubic interpolation and the new given time frequency.
New points are inserted between two original points if their time distance is longer than the new time frequency.

@param x_y_t_array: the array containing in each line the x and y coordinates and the timestamp of the sampling. x must contain
strictly increasing ordered values!
@param new_time_frequency: the new time frequency to interpolate points. Expressed in the same units as the sampling time stamps in the 3rd
column of the first parameter
@return: an array with the same structure, with new points inserted to respect the new sampling time frequency
@author: XXXXXX
"""
def timeSampleInterpolationPolinomial(x_y_t_array, new_time_frequency):
    if len(x_y_t_array) < 4: return None

    x = x_y_t_array[:, 0]
    y = x_y_t_array[:, 1]
    t = x_y_t_array[:, 2]

#    f = interp1d(x, y, kind='cubic')
    f = interp.PPoly(x, y)

    new_x_array = []
    new_t_array = []
    for i in range(1, x.size):
        delta_x = x[i] - x[i - 1]
        delta_t = t[i] - t[i - 1]

        new_x_array.append(x[i-1])
        new_t_array.append(t[i - 1])

        if delta_t > new_time_frequency:
            current_x_velocity = delta_x / delta_t
            actual_x = x[i - 1] + new_time_frequency * current_x_velocity
            actual_t = t[i-1] + new_time_frequency
            while actual_x < x[i] :
                new_x_array.append(actual_x)
                new_t_array.append(actual_t)
                actual_x += new_time_frequency * current_x_velocity
                actual_t += new_time_frequency

    new_x_array.append(x[:1])
    new_t_array.append(t[:1])

    new_y_array = f(new_x_array)

    result_array = np.column_stack((np.array(new_x_array), np.array(new_y_array), np.array(new_t_array)))

    return result_array

"""
Input : an array with point coordinates (x, y) first two columns and the timestamps for each point. Typically sampled mouse movements
The array has to contain at least 4 points! (interp1d constraint). The function completes this sample with points and timestamps using
spline interpolation and the new given time frequency.
New points are inserted between two original points if their time distance is longer than the new time frequency.
The program deletes consecutive rows containing identical x,y pairs

@param x_y_t_array: the array containing in each line the x and y coordinates and the timestamp of the sampling
@param new_time_frequency: the new time frequency to interpolate points. Expressed in the same units as the sampling time stamps in the 3rd
column of the first parameter
@return: an array with the same structure, wisth new points inserted to respect tne new sampling time frequence according to a
polinomial interpolation
@author: XXXXX
"""
def timeSampleInterpolationSpline(x_y_t_array, new_time_frequency):
    if len(x_y_t_array) < 4: return None
    x_y_t_array_f =np.array(x_y_t_array, dtype='float64')


    #delete  consecutive rows containing identical x,y pairs
    row_indees_to_delete = []
    for i in range (1,len(x_y_t_array_f)):
        if (x_y_t_array_f[i,0] == x_y_t_array_f[i-1,0]) and (x_y_t_array_f[i,1] == x_y_t_array_f[i-1,1]):
            row_indees_to_delete.append(i)
    x_y_t_array_f = np.delete(x_y_t_array_f,row_indees_to_delete,axis=0)

    x = x_y_t_array_f[:, 0]
    y = x_y_t_array_f[:, 1]
    t = x_y_t_array_f[:, 2]

    if len(x) < 4:
        return None
    tck, u = interp.splprep([x, y], s=1)

    norm_t = []
    norm_t = [(i -t[0]) for i in t]
    norm_t = [i / (t[-1] - t[0]) for i in norm_t]

    norm_new_frequency = new_time_frequency / (t[-1] - t[0])
    interpolation_points = []
    for i in range(1, len(norm_t)):
        low = (i - 1) / (len(norm_t) - 1)
        high = i / (len(norm_t) - 1)
        count = (norm_t[i] - norm_t[i - 1]) / norm_new_frequency
        temp_interpolation_points =[]
        if count>0:
            step = (high-low)/count;
            while low < high:
                temp_interpolation_points.append(low)
                low+=step
        else:
            temp_interpolation_points.append(low)
        interpolation_points.extend(temp_interpolation_points)

    interpolation_points.extend([1])

    out = interp.splev(interpolation_points, tck)

    new_time_points = [(i * (t[-1] - t[0])+t[0]) for i in interpolation_points]

    result_array = np.column_stack((np.array(out[0]), np.array(out[1]), np.array(new_time_points)))

    return result_array

"""
Input : an array with point coordinates (x, y) first two columns and the timestamps for each point. Typically sampled mouse movements
The array has to contain at least 2 points! The function completes this sample with points and timestamps using
spline interpolation and the new given time frequency.
New points are inserted between two original points if their time distance is longer than the new time frequency.
If new_time_frequency > the duration fo the sample, an empty array is returned.

@param x_y_t_array: the array containing in each line the x and y coordinates and the timestamp of the sampling
@param new_time_frequency: the new time frequency to interpolate points. Expressed in the same units as the sampling time stamps in the 3rd
column of the first parameter
@return: an array with the same structure, wisth new points inserted to respect tne new sampling time frequence in linear interpolation
@author: XXXXX
"""
def timeSampleInterpolationLinear(x_y_t_array, new_time_frequency):

    x_y_t_array_f = np.array(x_y_t_array, dtype='float64')
    x = x_y_t_array_f[:, 0]
    y = x_y_t_array_f[:, 1]
    t = x_y_t_array_f[:, 2]


    if new_time_frequency> (t[-1]-t[0]): return None
    new_x = []
    new_y = []
    new_t = []
    new_x.append(x[0])
    new_y.append(y[0])
    new_t.append(t[0])
    for i in range(1, len(t)):
        nb_points = int(np.floor((t[i]-t[i-1])/new_time_frequency))
        diff_x = x[i] - x[i - 1]
        diff_y = y[i] - y[i - 1]
        if nb_points>0:
            d_x = (x[i] - x[i - 1]) / nb_points
            for j in range(1,nb_points):
                new_x.append(x[i - 1]+j*d_x)
                if d_x!=0: new_y.append(new_y[-1]+(d_x*diff_y/diff_x))
                else: new_y.append(new_y[-1]+diff_y/nb_points)
                new_t.append(new_t[-1]+j*new_time_frequency)
        new_x.append(x[i])
        new_y.append(y[i])
        new_t.append(t[i])

    result_array = np.column_stack((np.array(new_x), np.array(new_y), np.array(new_t)))
    return result_array
