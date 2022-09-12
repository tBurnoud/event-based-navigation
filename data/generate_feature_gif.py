import cv2 as cv
import imageio
import dv_processing as dv
from feature_tracker import FeatureTracker
from contour_tracker import ContourTracker
import datetime
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy.spatial.transform import Rotation

#noise filter declaration and setting up its parameters
noiseFilter = dv.noise.FastDecayNoiseFilter((320, 240), subdivisionFactor=8, noiseThreshold=8)
noiseFilter.setHalfLife(datetime.timedelta(milliseconds=200))

proj = np.array([0., 0., 0.])

#value for depth linear interpolation
alpha = 0.5
prevDepth = 0

beta = 0.5

key = None

vis = dv.visualization.EventVisualizer((320, 240), (175, 175, 175))

feature_params = dict( maxCorners = 750,
                       qualityLevel = 0.5,
                       minDistance = 7,
                       blockSize = 7 )

clustering = DBSCAN(eps=35, min_samples=2)

clustersColor = [[255, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 255], 
                 [0, 255, 255], [125, 125, 125], [75, 75, 125], [125, 75, 75], [75, 125, 75], [0, 0, 0], [255, 255, 255]]


start_time = 0


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

#size of the drone used for object tracking (in meter) 
drone_size = (0.36, 0.14)

img_lst1 = []
img_lst2 = []
img_lst3 = []

def clustering_fct(pts, time):
    global key
    X = []
    for p in pts:
        # print(p[0])
        X.append([p[0][0], p[0][1]])
    df = pd.DataFrame(X)
    labels = clustering.fit_predict(df)

    output_img = np.zeros((240, 320, 3), dtype=np.uint8)

    #if the clustering found at least one label : 
    if labels is not None:
        #we loop on all labels found
        for i, p in enumerate(X):
            label = labels[i]
            color_index = label % len(clustersColor) - 1
            if label >=0:
                color = clustersColor[color_index]
            else:
                color = [0, 0, 255]
            cv.circle(output_img, (int(p[0]), int(p[1])), 1, color, 1)
        cv.imshow("DBSCAN", output_img)
        if time > 0.48 and time < 1.01 or time > 1.7 and time < 2.32:
            img_lst2.append(output_img)
        
        if key == 115:
            cv.imwrite("dbscan.png", output_img)

        return labels

def track(event_slice):
    global prevDepth
    global proj
    global key

    #processing the events through the event filter
    noiseFilter.accept(event_slice)
    filtered = noiseFilter.generateEvents()

    #getting the everage time of the current batch
    time = ((filtered.getLowestTime() + filtered.getHighestTime()) / 2e6) - start_time

    #recording was done with camera flipped upside down so we need to un-flip it first
    gray_ev = cv.flip(cv.flip(vis.generateImage(filtered), 0), 1)

    acc.accept(filtered)
    acc_img = cv.flip(cv.flip(acc.generateFrame().image, 0), 1)

    cv.imshow("acc", acc_img)

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

    evt_img = cv.flip(cv.flip(vis.generateImage(filtered), 0), 1)
    cv.imshow("Events", evt_img)

    corners =  np.zeros((240, 320, 3), dtype=np.uint8)

    #extracting the "corners" of the image
    pts = cv.goodFeaturesToTrack(cv.cvtColor(gray_ev, cv.COLOR_BGR2GRAY), mask = None, **feature_params)

    if pts is not None:
        labels = clustering_fct(pts, time)
        X = []
        for p in pts:
            X.append([p[0][0], p[0][1]])
            cv.circle(corners,(int(p[0][0]), int(p[0][1])), 1, (10, 10, 200))

        cv.imshow("corners", corners)

        df = pd.DataFrame(X)
        df  = pd.concat([df, pd.DataFrame(labels)], axis=1)
        df.columns = ['x', 'y', 'labels']
        for i in set(labels):
            if i != -1 and i < len(clustersColor):
                tmp = df.loc[df['labels'] == i]
                if not tmp.empty:
                    x_range = [int(np.min(tmp['x'])), int(np.max(tmp['x']))]
                    y_range = [int(np.min(tmp['y'])), int(np.max(tmp['y']))]
                    grey_ev = cv.rectangle(gray_ev, (x_range[1], y_range[1]), (x_range[0], y_range[0]), clustersColor[i])

                    size = (x_range[1] - x_range[0], y_range[1] - y_range[0])
                    if size[0] != 0 and size[1] != 0:
                        depth = ((F * drone_size[0]) / (size[0] * fxy)) + ((F * drone_size[1]) / (size[1] * fxy))
                        depth /= 2
                        prevDepth = alpha * depth + (1-alpha) * prevDepth
                        c = np.array([np.mean(x_range) - principalPoint[0], np.mean(y_range) - principalPoint[1], 1.0])

                        cX = (c.transpose() @ invMatrix) * 1
                        wX = extraMat @ np.array([cX[0], cX[1], prevDepth, 1.0])

                        proj = (wX * beta) + proj * (1-beta)

        # print(time)

        #used to save specific frames to gif
        # if time > 0.48 and time < 1.01 or time > 1.7 and time < 2.32:
        #     img_lst1.append(evt_img)
        #     img_lst3.append(gray_ev)
        
        cv.imshow("clustered", gray_ev)
        key = cv.waitKey(1)

        if key == 115:
            cv.imwrite("events.png", evt_img)
            cv.imwrite("grey.png", grey_ev)
            cv.imwrite("accumulated.png", acc_img)

if __name__ == "__main__":

    cv.namedWindow("Events", cv.WINDOW_NORMAL)
    cv.namedWindow("DBSCAN", cv.WINDOW_NORMAL)
    cv.namedWindow("clustered", cv.WINDOW_NORMAL)
    cv.namedWindow("acc", cv.WINDOW_NORMAL)

    filePath = "/home/tommy/workspace/dv-processing/data/exp1.aedat4"

    data = dv.io.MonoCameraRecording(filePath)

    # Initialize event accumulator with the known resolution
    acc = dv.Accumulator((320, 240))

    # Some accumulation parameters
    acc.setMaxPotential(1.0)

    slicer = dv.EventStreamSlicer()

    ft = FeatureTracker(sr=35, m_sample=2)

    batch = data.getNextEventBatch()

    start_time = (batch.getLowestTime() + batch.getHighestTime()) / 2e6

    while batch is not None:
        
        track(batch)

        batch = data.getNextEventBatch()

    imageio.mimsave("gif/jump/events.gif", img_lst1, fps=30)
    imageio.mimsave("gif/jump/DBSCAN.gif", img_lst2, fps=30)
    imageio.mimsave("gif/jump/feature_clus.gif", img_lst3, fps=30)
