from rosbag_reader import DataReader
from event_reader import EventReader, ClusteringType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

prefix = "/home/tommy/workspace/dv-processing/data/"

timeStep = 0.25

time_shift = 0.5

def time_convert(x):
    return (x-time_shift)*1e6

def convert_to_ros(coord):
    return np.array([coord[2], -coord[0]-0.119, -coord[1]])*4.2

def time_slice(df, t0, t1):
    # print(df.time)
    # print("slicing...", t0*1e6+start_time, t1*1e6+start_time)
    new_df = df[(df.time > t0*1e6+start_time) & (df.time < t1*1e6+start_time)] 
    return new_df

if __name__ == "__main__":

    #getting and formatting data from room tracking
    dr = DataReader(prefix + "bag_2022-07-11-16-10-42/mavros-local_position-pose.csv")
    ros_df = dr.get_data()
    ros_df['time'] = ros_df['time'].apply(time_convert)

    #getting and formatting data from event camera
    evr = EventReader(prefix + "exp1.aedat4")
    positions, exec_times = evr.run(downFactor=.75, threshold=100, clusterType=ClusteringType.CONTOUR)
    positions.columns = ["time", "x", "y", "z"]

    #initialise the plot used to display positions and error on position
    fig, axs = plt.subplots(3, 2)

    #getting the start and end time (in second) of the experiment 
    start_time = np.min(positions.time)
    mint = start_time / 1e6
    maxt = np.max(positions.time) / 1e6

    # print("time range ft", start_time, (mint-mint), (maxt-mint))

    prevT = 0

    #table used to store the error along a given axis, latter used for to compute average error
    errX = []
    errY = []
    errZ = []

    #tables use to store position value for both ground truth and estimation
    #used latter for correlation coefficients 
    arr1x  =[]
    arr1y = []
    arr1z = []
    arr2x = []
    arr2y = []
    arr2z = []

    ts = []

    print("starting comparison...")

    #loop from start to end time with an interval of timeStep (everything in second)
    for t in np.arange(0,(maxt - mint)  + timeStep, timeStep):
        
        #getting the subset corresponding to the current time-slice
        df1 = time_slice(ros_df, prevT, t)
        df2 = time_slice(positions, prevT, t)

        #getting the position from the tracking system for the current time slice
        pos1 = np.array([np.mean(df1.x), np.mean(df1.y), np.mean(df1.z)])

        #making sure that there is data in the current time-slice we're processing
        if len(df2) > 0:
            #getting the position estimation from the tested algorithm 
            pos2 = convert_to_ros(np.array([np.mean(df2.x), np.mean(df2.y), np.mean(df2.z)]))

            #computing the difference between ground truth and estimation
            diff = pos1 - pos2

            #if there is an estimate for the current time slice, 
            #we store the positions (ground truth and estimate) for the correlation coef
            if not np.isnan(np.sum(pos1)):
                arr1x.append(pos1[0])
                arr1y.append(pos1[1])
                arr1z.append(pos1[2])

                arr2x.append(pos2[0])
                arr2y.append(pos2[1])
                arr2z.append(pos2[2])

            #checking for nan
            if not np.isnan(np.sum(diff)):
                errX.append(diff[0])
                errY.append(diff[1])
                errZ.append(diff[2])

            #plotting part
            axs[0, 0].scatter(t, pos2[0], s=1, c='g')
            axs[0, 0].scatter(t, pos1[0], s=1, c='b')
            axs[0, 0].legend(["ground truth", "estimate"])
            axs[0, 0].set(xlabel="time", ylabel="x (m)")
            axs[0, 0].set_xlim([-0.1, 70.5])

            axs[1, 0].scatter(t, pos2[1], s=1, c='g')
            axs[1, 0].scatter(t, pos1[1], s=1, c='b')
            axs[1, 0].legend(["ground truth", "estimate"])
            axs[1, 0].set(xlabel="time", ylabel="y (m)")
            axs[1, 0].set_xlim([-0.1, 70.5])

            axs[2, 0].scatter(t, pos2[2], s=1, c='g')
            axs[2, 0].scatter(t, pos1[2], s=1, c='b')
            axs[2, 0].legend(["ground truth", "estimate"])
            axs[2, 0].set(xlabel="time", ylabel="z (m)")
            axs[2, 0].set_xlim([-0.1, 70.5])

            axs[0, 1].scatter(t, np.abs(diff[0]), s=1, c='b')
            axs[0, 1].set(xlabel="time", ylabel="x error")
            axs[0, 1].set_xlim([-0.1, 70.5])

            axs[1, 1].scatter(t, np.abs(diff[1]), s=1, c='b')
            axs[1, 1].set(xlabel="time", ylabel="y error")
            axs[1, 1].set_xlim([-0.1, 70.5])

            axs[2, 1].scatter(t, np.abs(diff[2]), s=1, c='b')
            axs[2, 1].set(xlabel="time", ylabel="z error")
            axs[2, 1].set_xlim([-0.1, 70.5])

        else:
            axs[0, 0].scatter(t, pos1[0], s=1, c='b')
            axs[1, 0].scatter(t, pos1[1], s=1, c='b')
            axs[2, 0].scatter(t, pos1[2], s=1, c='b')
            axs[0, 0].set_xlim([-0.1, 70.5])
            axs[1, 0].set_xlim([-0.1, 70.5])
            axs[2, 0].set_xlim([-0.1, 70.5])
            ts.append(t)

        prevT = t

    #computing average error on position for each axis
    mean_error = np.abs((np.mean(errX), np.mean(errY), np.mean(errZ)))
    median_error = np.abs((np.median(errX), np.median(errY), np.median(errZ)))
    tracking_time = (maxt - mint) - len(ts)*timeStep

    print("len(ts) :", len(ts))

    print("average error :", mean_error)
    print("median error :", median_error)
    print("max error:", np.max(np.abs(errX)), np.max(np.abs(errY)), np.max(np.abs(errZ)))
    print("min error:", np.min(np.abs(errX)), np.min(np.abs(errY)), np.min(np.abs(errZ)))
    print("tracked the drone for :", tracking_time, "/", maxt-mint, "s")

    #computing pearson correlation coefficient for each axis
    rx = scipy.stats.pearsonr(arr1x, arr2x)
    ry = scipy.stats.pearsonr(arr1y, arr2y)
    rz = scipy.stats.pearsonr(arr1z, arr2z)

    #computing spearman correlation coefficient for each axis
    sx = scipy.stats.spearmanr(arr1x, arr2x)
    sy = scipy.stats.spearmanr(arr1y, arr2y)
    sz = scipy.stats.spearmanr(arr1z, arr2z)

    #computing kendall correlation coefficient for each axis
    kx = scipy.stats.kendalltau(arr1x, arr2x)
    ky = scipy.stats.kendalltau(arr1y, arr2y)
    kz = scipy.stats.kendalltau(arr1z, arr2z)

    print("pearson correlation coeff :")
    print(rx)
    print(ry)
    print(rz)

    print("spearman correlation :")
    print(sx)
    print(sy)
    print(sz)

    print("kendall correlation")
    print(kx)
    print(ky)
    print(kz)

    plt.show()

    # X = np.arange(0,len(exec_times), 1)

    exec_times = np.array(exec_times) * 1e3

    fig, ax = plt.subplots(1,1)

    #TODO convert to ms and tweek histogram 

    ax.hist(exec_times, bins=50)
    plt.vlines(np.mean(exec_times), 0, 4000, color='red')
    # ax.scatter(X, exec_times)
    plt.show()

