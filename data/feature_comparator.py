from rosbag_reader import DataReader
from event_reader import EventReader, ClusteringType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

prefix = "/home/tommy/workspace/dv-processing/data/"

timeStep = 0.1

# start_time = 0

def time_convert(x):
    return (x-0.75)*1e6

def convert_to_ros(coord, scale=4.2):
    return np.array([coord[2], -coord[0]-0.0, -coord[1]])*scale

def time_slice(df, t0, t1):
    # print(df.time)
    # print("slicing...", t0*1e6+start_time, t1*1e6+start_time)
    new_df = df[(df.time > t0*1e6 + start_time) & (df.time < t1*1e6+start_time)] 
    return new_df

if __name__ == "__main__":

    #getting and formatting data from room tracking
    dr = DataReader(prefix + "bag_2022-07-11-16-10-42/mavros-local_position-pose.csv")
    ros_df = dr.get_data()
    ros_df['time'] = ros_df['time'].apply(time_convert)

    #initialising the event reader
    evr = EventReader(prefix + "exp1.aedat4")
    positions, exec_times = evr.run(clusterType=ClusteringType.FEATURE)
    positions.columns = ["time", "x", "y", "z"]

    #initialise the plot used to display positions and error on position
    fig, axs = plt.subplots(3, 3)

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

    #used to store mean error for tested params
    meansX = []
    meansY = []
    meansZ = []

    #used to store pearson correlations for each axis for tested params
    rx = []
    ry = []
    rz = []

    #used to store spearman correlations for each axis for tested params
    sx = []
    sy = []
    sz = []


    ts = []

    print("starting comparison...")

    #params list for search radius
    # step = 5
    # params = np.arange(10, 50+step, step)

    #params list for min sample
    step = 5
    params = np.arange(0, 20+step, step) 

    #looping on all parameters we want to test :
    for p in params:
        print("comparison done with :", p, "...")

        arr1x  =[]
        arr1y = []
        arr1z = []
        arr2x = []
        arr2y = []
        arr2z = []

        #running the event reader with the parameter of the current loop
        evr = EventReader(prefix + "exp1.aedat4")
        positions, exec_times = evr.run(sr=125, m_sample=p, clusterType=ClusteringType.FEATURE)
        if positions.empty:
            print("skipping...")
            continue
        positions.columns = ["time", "x", "y", "z"]

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
                pos2 = convert_to_ros(np.array([np.mean(df2.x), np.mean(df2.y), np.mean(df2.z)]), 4.2)

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

            
            else:
                ts.append(t)
            
            prevT = t

        rx.append(scipy.stats.pearsonr(arr1x, arr2x)[0])
        ry.append(scipy.stats.pearsonr(arr1y, arr2y)[0])
        rz.append(scipy.stats.pearsonr(arr1z, arr2z)[0])

        sx.append(scipy.stats.spearmanr(arr1x, arr2x)[0])
        sy.append(scipy.stats.spearmanr(arr1y, arr2y)[0])
        sz.append(scipy.stats.spearmanr(arr1z, arr2z)[0])
        
        meansX.append(np.mean(np.abs(errX)))
        meansY.append(np.mean(np.abs(errY)))
        meansZ.append(np.mean(np.abs(errZ)))

    axs[0, 0].scatter(params, meansX, s=3)
    # axs[0].bar(params, meansX, width=step)
    axs[0, 0].set_title("error on x-axis")

    axs[1, 0].scatter(params, meansY, s=3)
    # axs[1].bar(params, meansY, width=step)
    axs[1, 0].set_title("error on y-axis")

    axs[2, 0].scatter(params, meansZ, s=3)
    # axs[2].bar(params, meansZ, width=step)
    axs[2, 0].set_title("error on z-title")

    axs[0, 1].bar(params, rx, width=step/2)
    axs[0, 1].set_title("pearson correlations")

    axs[1, 1].bar(params, ry, width=step/2)
    axs[1, 1].set_title("pearson correlations")

    axs[2, 1].bar(params, rz, width=step/2)
    axs[2, 1].set_title("pearson correlations")

    axs[0, 2].bar(params, sx, width=step/2)
    axs[0, 2].set_title("spearman corrrelations")

    axs[1, 2].bar(params, sy, width=step/2)
    axs[1, 2].set_title("spearman corrrelations")
    
    axs[2, 2].bar(params, sz, width=step/2)
    axs[2, 2].set_title("spearman corrrelations")

    #computing average error on position for each axis
    mean_error = np.abs((np.mean(errX), np.mean(errY), np.mean(errZ)))
    median_error = np.abs((np.median(errX), np.median(errY), np.median(errZ)))
    tracking_time = (maxt - mint) - len(ts)*timeStep

    plt.show()

