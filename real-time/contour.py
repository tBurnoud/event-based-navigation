import dv_processing as dv
import cv2 as cv
import argparse
import time 
import numpy as np
import datetime
from sklearn.cluster import MeanShift, DBSCAN
import random as rng
import pandas as pd
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

clustersColor = [[255, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 255], 
                 [0, 255, 255], [125, 125, 125], [75, 75, 125], [125, 75, 75], [75, 125, 75]]

threshold = 250
downsizeFactor = 2/3
upscaleFactor = 1/downsizeFactor

f1 = dv.RefractoryPeriodFilter((320, 240))
f1.setRefractoryPeriod(datetime.timedelta(milliseconds=150))
f2 = dv.noise.BackgroundActivityNoiseFilter((320, 240))
f2.setBackgroundActivityDuration(datetime.timedelta(milliseconds=2000))

# clustering = MeanShift(bandwidth=None, bin_seeding=True, min_bin_freq=2, cluster_all=False, 
#                     #    max_iter=350, n_jobs=1)

clustering = DBSCAN(eps=20, min_samples=7)

DoClustering = False

x, y, w, h = 100, 100, 75, 75 # simply hardcoded the values
track_window = (x, y, w, h)

ballSize = 3.5
humanSize = 172

FOVx = 46.263494161722484
FOVy = 35.386736849945656
principalPoint = np.array([152.8244171142578, 116.64521026611328])
d = 400


fig = plt.figure()
ax = plt.axes(projection='3d')
# ax = plt.axes()
# camMatrix = np.array([[3.3028732685884881e2, 0., 1.6204341347407075e2],
#                       [0, 3.3214889523200645e2, 1.0989435392807357e2],
#                       [0., 0., 1.]])

camMatrix = np.array([[3.7453954539334285e+02, 0., 1.5282441586920595e+02],
                      [0, 3.7616000309566266e+02, 1.1664521377777844e+02],
                      [0., 0., 1.]])

invMatrix = np.linalg.inv(camMatrix)

fxy = 0.125#(camMatrix[0,0] + camMatrix[1, 1]) * 1e-4
F = 375e-3 #0.25

filter_chain = dv.EventFilterChain()
filter_chain.addFilter(f1)
filter_chain.addFilter(f2)

depths = []
bufferSize = 100

prevNbEvents = 0

def plot3Dproj(cX):
    ax.cla()
    ax.scatter(cX[0], cX[1], cX[2])
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    # plt.zlabel("z axis")
    plt.draw()
    plt.pause(0.5)

def plotDepth(depths):
    ax.cla()
    xs = range(len(depths))
    ax.scatter(xs, depths)
    plt.draw()
    plt.pause(0.25)


def apply_meanShift(img):
    global track_window
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    roi = img[y:y+h, x:x+w]
    hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    # hsv_roi = roi
    hsv_green = cv.cvtColor(np.uint8([[[0,255,0 ]]]),cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((0.,0.,100.)))
    mask2 = cv.inRange(hsv_roi, np.array((50., 255., 255.)), np.array((70.,255.,255.)))
    roi_hist = cv.calcHist([hsv_roi],[0],mask2,[180],[0,180])
        
    cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, 0)


    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv],[0],roi_hist,[0,500],1)

    # apply camshift to get the new location
    ret, track_window = cv.CamShift(gray, track_window, term_crit)

    # uncommment to display contours used for tracking
    # ret, track_window = cv.meanShift(gray, track_window, term_crit)
    # pts = cv.boxPoints(ret)
    # pts = np.int0(pts)
    # img2 = cv.polylines(img,[pts],True, 255,2)

    # cv.imshow("meanShift", img2)
    # cv.waitKey(2)

    return ret


def applyHough(event_slice):

    filter_chain.accept(event_slice)
    filtered = filter_chain.generateEvents()

    event_frame = acc.generateFrame()   
    acc.accept(event_slice)
    vis = dv.visualization.EventVisualizer((320, 240))
    events_img = cv.flip(cv.flip(vis.generateImage(event_slice), 0), 1)
    events_img = cv.resize(events_img, (int(320*downsizeFactor), int(240*downsizeFactor)))
    events_gray = cv.cvtColor(events_img, cv.COLOR_BGR2GRAY)
   
    # src = cv.flip(cv.flip(event_frame.image, 0), 1)
    src = events_img
    cdst = np.copy(src)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 7)

    cv.imshow("gray", gray)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 1,
                               param1=175, param2=30,
                               minRadius=2, maxRadius=150)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (255, 0, 0), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (0, 0, 255), 1)

    
    cv.namedWindow("detected circles", cv.WINDOW_NORMAL)
    cv.imshow("detected circles", src)
    cv.imshow("Events", events_img)
    cv.imshow("line detector", cdst)
    cv.waitKey(2)


def draw_contours(contours, canny_output):
    label = 0
    labels = None
    contours_poly = [None]*len(contours)
    # boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    X = []
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 5, True)
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
        if DoClustering:
            X.append([centers[i][0], centers[i][1]])

    if DoClustering:    
        df = pd.DataFrame(X)
        # df = df.sample(frac=0.5)
        if not df.empty:
            # print(df)
            labels = clustering.fit_predict(df)
    
    drawing = np.zeros((int(canny_output.shape[0]), int(canny_output.shape[1]), 3), dtype=np.uint8)
    
    
    for i in range(len(contours)):
        # color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        color = [200, 200, 200]
        if DoClustering and labels is not None:    
            label = labels[i]
            color_index = label % len(clustersColor) - 1
            if label >= 0:
                color = clustersColor[color_index]
            else:
                color = [0, 0, 255]

        #uncommment to display bonding rect
        # drawing = cv.drawContours(drawing, contours_poly, i, color)
        # cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #   (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        # cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 1)
        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), 1, color, 1)

    if DoClustering:
        cv.imshow("contours", drawing)
        cv.waitKey(2)
    return drawing

def Unproject(points, Z, intrinsic, distortion):
  f_x = intrinsic[0, 0]
  f_y = intrinsic[1, 1]
  c_x = intrinsic[0, 2]
  c_y = intrinsic[1, 2]

  # Step 1. Undistort.
  points_undistorted = np.array([])
  if len(points) > 0:
    points_undistorted = cv.undistortPoints(points, intrinsic, distortion, P=intrinsic)
    points_undistorted = np.squeeze(points_undistorted, axis=1)

  # Step 2. Reproject.
  result = []
  for idx in range(points_undistorted.shape[0]):
    z = Z#Z[0] if len(Z) == 1 else Z[idx]
    x = (points_undistorted[idx, 0] - c_x) / f_x * z
    y = (points_undistorted[idx, 1] - c_y) / f_y * z
    result.append([x, y, z])
  return result

def find_contour(event_slice):
    global prevNbEvents
    global depths
    global track_window
    filter_chain.accept(event_slice)
    filtered = filter_chain.generateEvents()
    filtered = event_slice
    distortion = np.array([-0.6678174138069153, 2.238603115081787, 0.0018292812164872885, 0.0028341736178845167, -4.129987716674805])
    
    event_frame = acc.generateFrame()   
    acc.accept(filtered)
    vis = dv.visualization.EventVisualizer((320, 240))
    events_img = cv.flip(cv.flip(vis.generateImage(filtered), 0), 1)
    events_img = cv.resize(events_img, (int(320*downsizeFactor), int(240*downsizeFactor)))

    img = cv.flip(cv.flip(event_frame.image, 0), 1)
    img = cv.resize(img, (int(320*downsizeFactor), int(240*downsizeFactor)))

    canny_input = events_img
    canny_output = cv.Canny(canny_input, threshold, threshold*2)

    
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if not DoClustering:
        boundingBox = apply_meanShift(canny_output)
        pts = cv.boxPoints(boundingBox)
        pts = np.int0(pts)
        cv.polylines(canny_input,[pts],True, 255,2)
        area = boundingBox[1][0] * boundingBox[1][1]
        py = []
        for p in pts:
            py.append(p[1])
        min_y = np.min(py)
        max_y = np.max(py)
        size = max_y - min_y
        
        if size != 0 and area > 300:
            depth = (F * ballSize) / (size * fxy)
            humanDepth = (F * humanSize) / (size * fxy)
            depths.append(humanDepth)
            c = np.array([(boundingBox[0][0] - principalPoint[0]) / camMatrix[0,0], (boundingBox[0][1] - principalPoint[1]) / camMatrix[1,1], 1])

            cX = (invMatrix @ c.transpose()) * depth
            c1 = np.array([c[0], c[1]])
            unprojectPoint = Unproject(c1, depth, camMatrix, distortion)  
        else:
            track_window = (50, 50, track_window[2], track_window[3])

    cv.imshow("Events", canny_input)
    cv.waitKey(2)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Attempt at camshift on an event stream.')
    args = parser.parse_args()

    camera = dv.io.CameraCapture()

    acc = dv.Accumulator(camera.getEventResolution())
    
    # Some accumulation parameters
    acc.setMaxPotential(1.0)
    acc.setEventContribution(0.75)
    acc.setRectifyPolarity(False)

    # Create the preview window
    # cv.namedWindow("Events", cv.WINDOW_NORMAL)
    # cv.namedWindow("depth", cv.WINDOW_NORMAL)
    # cv.namedWindow("Canny", cv.WINDOW_NORMAL)


    # Create an event slicer, this will only be used events only camera
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=15), find_contour)
        
    # start read loop
    while True:
        # Get events
        events = camera.getNextEventBatch()

        # If no events arrived yet, continue reading
        if events is None:
            continue

        slicer.accept(events)

