import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import bagpy
from bagpy import bagreader

class DataInit:
    def __init__(self):
        b = bagreader("/home/tommy/workspace/dv-processing/data/bag_2022-07-11-16-10-42.bag")

        print(b.topic_table)

        b.message_by_topic("/mavros/local_position/pose")


class DataReader:
    def __init__(self, filePath):
        #loading data from file
        print("loading data from ", filePath)
        data = pd.read_csv(filePath)

        #removing some part of the data where nothing happen
        self.df = data[(data.Time > 1657548725.4) & (data.Time < 1657548796.2)]
        # self.df = data

        #load some value from data into tables
        self.T = pd.DataFrame(self.df['Time'])
        self.X = pd.DataFrame(self.df['pose.position.x'])
        self.Y = pd.DataFrame(self.df['pose.position.y'])
        self.Z = pd.DataFrame(self.df['pose.position.z'])

        # print(self.X)
        self.table = self.T.join(self.X)
        self.table = self.table.join(self.Y)
        self.table = self.table.join(self.Z)
        self.table.columns = ["time", "x", "y", "z"]
        # self.table = np.array([self.T, self.X, self.Y, self.Z])

        # print(self.table)
    
    def plot(self):
        self.fig = plt.figure()
        self.ax = plt.axes()
        plt.scatter(self.T, self.X, s=1)
        plt.scatter(self.T, self.Y, s=1)
        plt.scatter(self.T, self.Z, s=1)
        plt.legend(["x", "y", "z"])
        plt.show()

    def shape(self):
        return self.table.shape

    def time_range(self):
        mint = np.min(self.T)
        maxt = np.max(self.T)

        return (mint-mint, maxt-mint)
    
    def get_data(self):
        return self.table
    



#uncomment to read bag file and generate csv
# di = DataInit()

#uncomment to read csv file and plot data
# dr = DataReader("/home/tommy/workspace/dv-processing/data/bag_2022-07-11-16-10-42/mavros-local_position-pose.csv")
# dr.plot()