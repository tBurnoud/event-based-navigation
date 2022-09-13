from IPython.display import display, clear_output
from PIL import Image
import dv_processing as dv
import time
import numpy as np
import datetime
import cv2 as cv
import random
import math

noiseFilter = dv.noise.FastDecayNoiseFilter((320, 240), subdivisionFactor=8, noiseThreshold=2)
noiseFilter.setHalfLife(datetime.timedelta(milliseconds=30))


x, y, w, h = 100, 100, 75, 75 # simply hardcoded the values
track_window = (x, y, w, h)

ballSize = 3.5
humanSize = 170


camMatrix = np.array([[3.7453954539334285e+02, 0., 1.5282441586920595e+02],
                      [0, 3.7616000309566266e+02, 1.1664521377777844e+02],
                      [0., 0., 1.]])

invMatrix = np.linalg.inv(camMatrix)

principalPoint = np.array([152.8244171142578, 116.64521026611328])
fxy = 0.08
F = 375e-3 

prev_frame = None

nbCorner = 300

color = np.random.randint(0, 255, (nbCorner+1, 3))

# params for corner detection
feature_params = dict( maxCorners = nbCorner,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
  
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                              10, 0.03))

def apply_meanShift(img):
    global track_window
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    roi = img[y:y+h, x:x+w]
    hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
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

    #uncomment to display bonding box
    # pts = cv.boxPoints(ret)
    # pts = np.int0(pts)
    # img2 = cv.polylines(img,[pts],True, 255,2)
    # cv.imshow("meanShift", img2)
    # cv.waitKey(2)

    return ret

def lk(event_slice):
    global prev_frame
    global prev_pts
    global color

    filtered = noiseFilter.generateEvents()
    noiseFilter.accept(event_slice)

    vis = dv.visualization.EventVisualizer((320, 240), (175, 175, 175))

    event_frame = acc.generateFrame()
    acc.accept(filtered)
    gray_ev = cv.flip(cv.flip(vis.generateImage(filtered), 0), 1)
    gray = cv.flip(cv.flip(event_frame.image, 0), 1)
    grey = gray_ev
    pts = cv.goodFeaturesToTrack(cv.cvtColor(gray_ev, cv.COLOR_BGR2GRAY), mask = None, **feature_params)


    if prev_frame is not None and pts is not None:
        mask = np.zeros_like(prev_frame)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame, gray, pts, None, **lk_params)
        good_new = p1[st == 1]
        good_old = pts[st == 1]

        lk = np.zeros((240, 320, 3), dtype = "uint8")
        
        V = np.array([0., 0.])
        i = 0
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            if err[i] > 5:
                a, b = new.ravel()
                c, d = old.ravel()
                A = np.array([a, b])
                B = np.array([c, d])
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                v = np.array([B - A])
                V = V + v
                i += 1

                mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)

                lk = cv.circle(lk, (a, b), 5, color[i].tolist(), -1)
        V /= i



        img = cv.add(lk, mask)

        boundingBox = apply_meanShift(lk)
        bPts = cv.boxPoints(boundingBox)
        bPts = np.int0(bPts)
        cv.polylines(grey,[bPts],True, 255,2)

        area = boundingBox[1][0] * boundingBox[1][1]
        py = []
        for p in bPts:
            py.append(p[1])
        min_y = np.min(py)
        max_y = np.max(py)
        size = max_y - min_y

        if size != 0 and area > 300:
            depth = (F * ballSize) / (size * fxy)
            humanDepth = (F * humanSize) / (size * fxy)
            print("ball\t", depth * 100, "cm")
            print("human\t", humanDepth , "m\n")
            c = np.array([(boundingBox[0][0] - principalPoint[0]) / camMatrix[0,0], (boundingBox[0][1] - principalPoint[1]) / camMatrix[1,1], 1])
            cX = (invMatrix @ c.transpose()) * depth

        # cv.imshow("tracker", grey)
        # cv.imshow("img", img)
    

        prev_pts = pts
    
    prev_frame = gray
    cv.waitKey(2)

def tracking(event_slice):
    noiseFilter.accept(event_slice)
    filtered = noiseFilter.generateEvents()

    tk = dv.TimedKeyPoint((160, 120), 50, -1, 0., 1, 1, 0)

    event_frame = acc.generateFrame()
    acc.accept(filtered)
    gray = cv.flip(cv.flip(event_frame.image, 0), 1)
    img = cv.cvtColor(gray,cv.COLOR_GRAY2RGB)

    config = dv.features.LucasKanadeConfig()
    config.searchWindowSize = (10, 10)
    config.numPyrLayers = 4
    config.maskedFeatureDetect = True
    tracker = dv.features.EventFeatureLKTracker.RegularTracker((320, 240), config)
    tracker.setNumberOfEvents(25)
    tracks = dv.features.FeatureTracks()
    tracker.accept(filtered)
    tracks.setHistoryDuration(datetime.timedelta(milliseconds=5000))

    results = tracker.runTracking()
    tracker_img = tracker.getAccumulatedFrame()
    if results is not None:
        tracks.accept(results)


    if tracker_img is not None:
        tracker_img = tracks.visualize(tracker_img)
        tracker_img = cv.flip(cv.flip(tracker_img, 0), 1)
        cv.imshow("tracker", tracker_img)
        time.sleep(0.005)
        clear_output(wait=True)

    cv.imshow("Accumulated events", img)
    cv.waitKey(2)

if __name__ == "__main__":
    camera = dv.io.CameraCapture()

    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=50), lk)

    acc = dv.Accumulator(camera.getEventResolution())
    
    acc.setMaxPotential(1.0)
    acc.setEventContribution(0.12)
    acc.setRectifyPolarity(False)

    prev_pts = np.array([[random.randint(0, 320), random.randint(0, 240)]])
    for i in range(9):
        tmp = np.array([[random.randint(0, 320), random.randint(0, 240)]])
        prev_pts = np.append(prev_pts, tmp, axis=0)

    cv.namedWindow("tracker", cv.WINDOW_NORMAL)
    cv.namedWindow("img", cv.WINDOW_NORMAL)

    while True:
        events = camera.getNextEventBatch()

        if events is None:
            continue
        
        slicer.accept(events)