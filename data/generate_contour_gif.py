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

threshold = 165

downsizeFactor = 1

img_lst1 = []
img_lst2 = []
img_lst3 = []

x, y, w, h = 160, 120, 30, 30 # simply hardcoded the values
track_window = (x, y, w, h)

def apply_meanShift(img, time):
    global track_window
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    roi = img[y:y+h, x:x+w]
    hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    hsv_green = cv.cvtColor(np.uint8([[[0,255,0 ]]]),cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((0.,0.,100.)))
    mask2 = cv.inRange(hsv_roi, np.array((50., 255., 255.)), np.array((70.,255.,255.)))
    roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
        
    cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv],[0],roi_hist,[0,500],1)

    # apply camshift to get the new location
    ret, track_window = cv.CamShift(gray, track_window, term_crit)
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    cv.polylines(img,[pts],True, 255,2)
    
    if time > 21 and time < 25:
        img_lst2.append(img)
    return ret

def track(event_slice):
    global prevDepth
    global proj
    #processing the events through the event filter
    noiseFilter.accept(event_slice)
    filtered = noiseFilter.generateEvents()

    #getting the everage time of the current batch
    time = ((filtered.getLowestTime() + filtered.getHighestTime()) / 2e6) - start_time

    #recording was done with camera flipped upside down so we need to un-flip it first
    img = cv.flip(cv.flip(vis.generateImage(filtered), 0), 1)

    events_img = cv.flip(cv.flip(vis.generateImage(filtered), 0), 1)

    black = np.zeros((240, 320, 3), dtype=np.uint8) + (175, 175, 175)


    a = int(20*downsizeFactor)
    b =int(320*downsizeFactor)
    ROI = black[0:a, 0:b]
    img[0:a, 0:b] = ROI

    a = int(172*downsizeFactor)
    b = int(182*downsizeFactor)
    c = int(220*downsizeFactor)
    d = int(230*downsizeFactor)
    ROI = black[a:b, c:d]
    img[a:b, c:d] = ROI

    cv.imshow("img", img)

    if time > 21 and time < 25:
        img_lst1.append(events_img)

    canny_input = img
    canny_output = cv.Canny(canny_input, threshold, threshold*2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.imshow("contour", canny_output)

    boundingBox = apply_meanShift(canny_output, time)
    pts = cv.boxPoints(boundingBox)
    pts = np.int0(pts)
    cv.polylines(canny_input,[pts],True, 255,2)
    area = boundingBox[1][0] * boundingBox[1][1]
    px = []
    py = []
    for p in pts:
        px.append(p[0])
        py.append(p[1])
    min_x = np.min(px)
    max_x = np.max(px)
    min_y = np.min(py)
    max_y = np.max(py)

    size = (boundingBox[1][1], boundingBox[1][0])

    if time > 21 and time < 25:
        img_lst3.append(canny_input)

    if area > 50:
        depth = ((F * drone_size[0]) / (size[0] * fxy)) + ((F * drone_size[1]) / (size[1] * fxy))
        depth /= 2


        prevDepth = alpha * depth + (1-alpha) * prevDepth
        c = np.array([boundingBox[0][0] - principalPoint[0], boundingBox[0][1] - principalPoint[1], 1.0])

        cX = (c.transpose() @ invMatrix) * 1
        wX = extraMat @ np.array([cX[0], cX[1], prevDepth, 1.0])
        proj = (wX * beta) + proj * (1-beta)


        key = cv.waitKey(1)
        if key == 115:
            cv.imwrite("events.png", gray_ev)
            cv.imwrite("contour.png", contours)

if __name__ == "__main__":

    cv.namedWindow("contour", cv.WINDOW_NORMAL)
    cv.namedWindow("img", cv.WINDOW_NORMAL)

    filePath = "/home/tommy/workspace/dv-processing/data/exp1.aedat4"

    data = dv.io.MonoCameraRecording(filePath)

    slicer = dv.EventStreamSlicer()

    ft = FeatureTracker(sr=35, m_sample=2)

    batch = data.getNextEventBatch()

    start_time = (batch.getLowestTime() + batch.getHighestTime()) / 2e6

    while batch is not None:
        
        track(batch)

        batch = data.getNextEventBatch()

    imageio.mimsave("gif/contour/events.gif", img_lst1, fps=30)
    imageio.mimsave("gif/contour/DBSCAN.gif", img_lst2, fps=30)
    imageio.mimsave("gif/contour/feature_clus.gif", img_lst3, fps=30)
