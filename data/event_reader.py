import dv_processing as dv
import pandas as pd
import cv2 as cv
from enum import Enum
import time

from feature_tracker import FeatureTracker
from contour_tracker import ContourTracker


prefix = "/home/tommy/workspace/stage/data/"

class ClusteringType(Enum):
    FEATURE = 1
    CONTOUR = 2

class EventReader:
    def __init__(self, filePath):

        print("opening ", filePath, "...")

        #loading data recorded by event camera
        self.data = dv.io.MonoCameraRecording(filePath)

        #initialise slicer
        self.slicer = dv.EventStreamSlicer()

        self.events = []

        self.acc = dv.Accumulator(self.data.getEventResolution())
        self.acc.setMaxPotential(1.0)
        self.acc.setEventContribution(0.12)


    def visualise_events(self, event_slice):
        self.acc.accept(event_slice)
        frame = self.acc.generateFrame()


        cv.imshow("accumulator", frame.image)
        cv.waitKey(2)

    def getNextEventBatch(self):
        return self.data.getNextEventBatch()



    def generate_events(self):
        
        print("loading data...")
        i = 0


        eventStore = self.data.getNextEventBatch()
        while eventStore is not None:
            for ev in eventStore:
                tmp = [ev.time(), ev.x(), ev.y(), ev.polarity()]
                self.events.append(tmp)
            eventStore = self.data.getNextEventBatch()
        
        df = pd.DataFrame(self.events)
        return df
    
    def run(self, sr=30, m_sample=3, downFactor=0.66, threshold=165, clusterType=ClusteringType.FEATURE):
        # cv.namedWindow("ev", cv.WINDOW_NORMAL)
        # cv.namedWindow("DBSCAN", cv.WINDOW_NORMAL)

        print("clustering type :", clusterType)

        exec_times = []

        if clusterType == ClusteringType.FEATURE:
            ft = FeatureTracker(sr, m_sample)
        elif clusterType == ClusteringType.CONTOUR:
            ft = ContourTracker(downFactor=downFactor, threshold=threshold)
        batch = self.getNextEventBatch()

        start_time = (batch.getLowestTime() + batch.getHighestTime()) / 2

        while batch is not None:
            t0 = time.time()
            ft.track(batch)
            t1 = time.time()
            exec_times.append(t1-t0)
            current_time = (batch.getLowestTime() + batch.getHighestTime()) / 2
            # print((current_time - start_time)/1e6)
            batch = self.getNextEventBatch()
        print("end of clustering")

        return ft.get_positions(), exec_times
