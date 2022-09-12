import dv_processing as dv
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
import cv2 as cv
import datetime
import pandas as pd
import numpy as np
import math


clustersColor = [[255, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 255], 
                 [0, 255, 255], [125, 125, 125], [75, 75, 125], [125, 75, 75], [75, 125, 75], [0, 0, 75],
                 [0, 75, 0], [75, 0, 0]]




#camera intrinsict matrix
camMatrix = np.array([[3.7453954539334285e+02, 0., 1.5282441586920595e+02],
                      [0, 3.7616000309566266e+02, 1.1664521377777844e+02],
                      [0., 0., 1.]])


camTransfo = np.array([[0.], [0.], [0.]])

#camera rotation
camRota = Rotation.from_euler('y', -20, degrees=True).as_matrix()

#extrinsict camera matrix
extraMat = np.hstack((camRota, camTransfo)) 

invMatrix = np.linalg.inv(camMatrix)


#camera parameters 
principalPoint = np.array([160, 120])
fxy = 0.04
F = 375e-3 

drone_size = (0.36, 0.14)

class FeatureTracker:
    def __init__(self, sr=45, m_sample=2):

        self.position = []

        self.proj = np.array([0., 0., 0.])
        self.beta = 0.75

        #clustering algorithm initialisation
        self.clustering = DBSCAN(eps=sr, min_samples=m_sample)

        self.noiseFilter = dv.noise.FastDecayNoiseFilter((320, 240), subdivisionFactor=8, noiseThreshold=8)
        self.noiseFilter.setHalfLife(datetime.timedelta(milliseconds=1000))

        #event accumulator
        self.acc = dv.Accumulator((320, 240))
        self.acc.setMaxPotential(1.0)
        self.acc.setEventContribution(0.12)

        #event visualiaser
        self.vis = dv.visualization.EventVisualizer((320, 240), (175, 175, 175))

        # params for corner detection
        self.feature_params = dict( maxCorners = 750,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

        self.prevDepth = 0
        self.alpha = 0.8

    def clustering_fct(self, pts):
        X = []
        for p in pts:
            X.append([p[0][0], p[0][1]])
        df = pd.DataFrame(X)
        labels = self.clustering.fit_predict(df)
        output_img = np.zeros((240, 320, 3), dtype=np.uint8)

        if labels is not None:
            for i, p in enumerate(X):
                label = labels[i]
                color_index = label % len(clustersColor) - 1
                if label >=0:
                    color = clustersColor[color_index]
                else:
                    color = [0, 0, 255]
                cv.circle(output_img, (int(p[0]), int(p[1])), 1, color, 1)

            # cv.imshow("DBSCAN", output_img)
            # cv.waitKey(0)
            return labels

    def track(self, event_slice):
        #processing the events through the event filter
        self.noiseFilter.accept(event_slice)
        filtered = self.noiseFilter.generateEvents()

        #getting the everage time of the current batch
        time = (filtered.getLowestTime() + filtered.getHighestTime()) / 2

        #recording was done with camera flipped upside down so we need to un-flip it first
        gray_ev = cv.flip(cv.flip(self.vis.generateImage(filtered), 0), 1)

        black = np.zeros((240, 320, 3), dtype=np.uint8) + (175, 175, 175)
        
        a = int(220)
        b = int(230)
        c = int(172)
        d = int(182)
        ROI = black[c:d, a:b]
        gray_ev[c:d, a:b] = ROI

        a = 20
        b = 320
        ROI = black[0:a, 0:b]
        gray_ev[0:a, 0:b] = ROI

        #extracting the "corners" of the image
        pts = cv.goodFeaturesToTrack(cv.cvtColor(gray_ev, cv.COLOR_BGR2GRAY), mask = None, **self.feature_params)

        if pts is not None:
            labels = self.clustering_fct(pts)
            X = []
            for p in pts:
                X.append([p[0][0], p[0][1]])

            df = pd.DataFrame(X)
            df  = pd.concat([df, pd.DataFrame(labels)], axis=1)
            df.columns = ['x', 'y', 'labels']
            for i in set(labels):
                if i != -1:
                    tmp = df.loc[df['labels'] == i]
                    if not tmp.empty:
                        x_range = [int(np.min(tmp['x'])), int(np.max(tmp['x']))]
                        y_range = [int(np.min(tmp['y'])), int(np.max(tmp['y']))]
                        cv.rectangle(gray_ev, (x_range[1], y_range[1]), (x_range[0], y_range[0]), clustersColor[i])

                        size = (x_range[1] - x_range[0], y_range[1] - y_range[0])
                        if size[0] != 0 and size[1] != 0:
                            depth = (F * drone_size[0]) / (size[0] * fxy)
                            prevDepth = self.alpha * depth + (1-self.alpha) * self.prevDepth
                            c = np.array([np.mean(x_range) - principalPoint[0], np.mean(y_range) - principalPoint[1], 1.0])

                            cX = (c.transpose() @ invMatrix) * 1
                            wX = extraMat @ np.array([cX[0], cX[1], prevDepth, 1.0])

                            self.proj = (wX * self.beta) + self.proj * (1-self.beta)

                            if i == 0:
                                self.position.append([time, self.proj[0], self.proj[1], self.proj[2]])
                                cv.circle(gray_ev, (int(c[0]+principalPoint[0]), int(c[1]+principalPoint[1])), 10, (10, 10, 225))

            # cv.imshow("ev", gray_ev)
            # cv.waitKey(2)

    def get_positions(self):
        return pd.DataFrame(self.position)

    def write_position(self):
        df = pd.DataFrame(self.position)
        df.columns = ["time", "x", "y", "z"]
        if not df.empty:
            df.to_csv("/home/tommy/workspace/dv-processing/data/feature_position.csv")
